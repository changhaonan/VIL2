import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from vil2.utils.pointnet import farthest_point_sample, index_points, square_distance


class DropoutSampler(torch.nn.Module):
    def __init__(self, num_features, num_outputs, dropout_rate = 0.5):
        super(DropoutSampler, self).__init__()
        self.linear = nn.Linear(num_features, num_features)
        self.linear2 = nn.Linear(num_features, num_features)
        self.predict = nn.Linear(num_features, num_outputs)
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = F.relu(self.linear(x))
        if self.dropout_rate > 0:
            x = F.dropout(x, self.dropout_rate)
        x = F.relu(self.linear2(x))
        return self.predict(x)
    
class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    
class StackedAttention(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        
        x = torch.cat((x1, x2), dim=1)

        return x
    
def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 
    
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]

    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)

    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points

class PointTransformerEncoderLarge(nn.Module):
    def __init__(self, output_dim=256, input_dim=6, mean_center=True):
        super(PointTransformerEncoderLarge, self).__init__()

        self.mean_center = mean_center

        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = StackedAttention(channels=256)

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(768, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)

    def forward(self, xyz, f):
        # xyz: B, N, 3
        # f: B, N, D
        center = torch.mean(xyz, dim=1)
        if self.mean_center:
            xyz = xyz - center.view(-1, 1, 3).repeat(1, xyz.shape[1], 1)
        x = self.pct(torch.cat([xyz, f], dim=2))  # B, output_dim

        return center, x

    def pct(self, x):

        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, D, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)

        return x
class Pooling(torch.nn.Module):
	def __init__(self, pool_type='max'):
		self.pool_type = pool_type
		super(Pooling, self).__init__()

	def forward(self, input):
		if self.pool_type == 'max':
			return torch.max(input, 2)[0].contiguous()
		elif self.pool_type == 'avg' or self.pool_type == 'average':
			return torch.mean(input, 2).contiguous()
		elif self.pool_type == 'sum':
			return torch.sum(input, 2).contiguous()
class PointNet(torch.nn.Module):
	def __init__(self, emb_dims=1024, input_shape="bnc", use_bn=False, global_feat=True):
		# emb_dims:			Embedding Dimensions for PointNet.
		# input_shape:		Shape of Input Point Cloud (b: batch, n: no of points, c: channels)
		super(PointNet, self).__init__()
		if input_shape not in ["bcn", "bnc"]:
			raise ValueError("Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' ")
		self.input_shape = input_shape
		self.emb_dims = emb_dims
		self.use_bn = use_bn
		self.global_feat = global_feat
		self.pooling = Pooling('max')

		self.layers = self.create_structure()

	def create_structure(self):
		self.conv1 = torch.nn.Conv1d(3, 64, 1)
		self.conv2 = torch.nn.Conv1d(64, 64, 1)
		self.conv3 = torch.nn.Conv1d(64, 64, 1)
		self.conv4 = torch.nn.Conv1d(64, 128, 1)
		self.conv5 = torch.nn.Conv1d(128, self.emb_dims, 1)
		self.relu = torch.nn.ReLU()

		if self.use_bn:
			self.bn1 = torch.nn.BatchNorm1d(64)
			self.bn2 = torch.nn.BatchNorm1d(64)
			self.bn3 = torch.nn.BatchNorm1d(64)
			self.bn4 = torch.nn.BatchNorm1d(128)
			self.bn5 = torch.nn.BatchNorm1d(self.emb_dims)

		if self.use_bn:
			layers = [self.conv1, self.bn1, self.relu,
					  self.conv2, self.bn2, self.relu,
					  self.conv3, self.bn3, self.relu,
					  self.conv4, self.bn4, self.relu,
					  self.conv5, self.bn5, self.relu]
		else:
			layers = [self.conv1, self.relu,
					  self.conv2, self.relu, 
					  self.conv3, self.relu,
					  self.conv4, self.relu,
					  self.conv5, self.relu]
		return layers
	def forward(self, input_data):
		# input_data: 		Point Cloud having shape input_shape.
		# output:			PointNet features (Batch x emb_dims)
		if self.input_shape == "bnc":
			num_points = input_data.shape[1]
			center = torch.mean(input_data, dim=1)
			input_data = input_data.permute(0, 2, 1)
			
		else:
			num_points = input_data.shape[2]
		if input_data.shape[1] != 3:
			raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

		output = input_data
		for idx, layer in enumerate(self.layers):
			output = layer(output)
		
		if self.global_feat:
		    return center, self.pooling(output)

class PointTransformerEncoderSmall(nn.Module):

    def __init__(self, output_dim=256, input_dim=6, mean_center=True):
        super(PointTransformerEncoderSmall, self).__init__()

        self.mean_center = mean_center

        # map the second dim of the input from input_dim to 64
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=64)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=64)
        self.pt_last = StackedAttention(channels=64)

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(192, 256, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(256),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(256, 256, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(256, 256)

    def forward(self, xyz, f=None):
        # xyz: B, N, 3
        # f: B, N, D
        center = torch.mean(xyz, dim=1)
        # print("center:", center[0])
        if self.mean_center:
            xyz = xyz - center.view(-1, 1, 3).repeat(1, xyz.shape[1], 1)
        if f is None:
            x = self.pct(xyz)
        else:
            x = self.pct(torch.cat([xyz, f], dim=2))  # B, output_dim

        return center, x

    def pct(self, x):

        # x: B, N, D
        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=128, nsample=32, xyz=xyz, points=x)
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)  # B, nsamples, D
        new_xyz, new_feature = sample_and_group(npoint=32, nsample=16, xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature)  # B, D, nsamples

        x = self.pt_last(feature_1)  # B, D * 2, nsamples
        x = torch.cat([x, feature_1], dim=1)  # B, D * 3, nsamples
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)

        return x
    
class EncoderMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, pt_dim=3, uses_pt=True):
        super(EncoderMLP, self).__init__()
        self.uses_pt = uses_pt
        self.output = out_dim
        d5 = int(in_dim)
        d6 = int(2 * self.output)
        d7 = self.output
        self.encode_position = nn.Sequential(
                nn.Linear(pt_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.ReLU(),
                )
        d5 = 2 * in_dim if self.uses_pt else in_dim
        self.fc_block = nn.Sequential(
            nn.Linear(int(d5), d6),
            nn.LayerNorm(int(d6)),
            nn.ReLU(),
            nn.Linear(int(d6), d6),
            nn.LayerNorm(int(d6)),
            nn.ReLU(),
            nn.Linear(d6, d7))

    def forward(self, x, pt=None):
        if self.uses_pt:
            if pt is None: raise RuntimeError('did not provide pt')
            y = self.encode_position(pt)
            x = torch.cat([x, y], dim=-1)
        return self.fc_block(x)
