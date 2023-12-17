"""Modified from official Diffusion Policy"""
from __future__ import annotations
import diffusers
import collections
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from vil2.data.obj_dp_dataset import normalize_data, unnormalize_data
import vil2.utils.torch_utils as torch_utils
import clip
from vil2.utils.eval_utils import compare_distribution
from vil2.data_gen.data_loader import visualize_pcd_with_open3d


class DmorpModel:
    """Diffusion Model for multi-object relative Pose Generation"""

    def __init__(self, cfg, vision_encoder, noise_pred_net, device):
        self.cfg = cfg
        # check cuda and mps
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
        # self.device = "cpu"
        # parameters
        self.max_scene_size = cfg.MODEL.MAX_SCENE_SIZE
        self.action_dim = cfg.MODEL.NOISE_NET.INIT_ARGS["input_dim"]
        self.pose_dim = cfg.MODEL.POSE_DIM
        self.time_emb_dim = cfg.MODEL.TIME_EMB_DIM if cfg.MODEL.USE_POSITIONAL_EMBEDDING else 1
        self.geometry_feat_dim = cfg.MODEL.GEOMETRY_FEAT_DIM
        self.semantic_feat_dim = cfg.MODEL.SEMANTIC_FEAT_DIM
        self.use_positional_embedding = cfg.MODEL.USE_POSITIONAL_EMBEDDING
        self.aggregate_type = cfg.MODEL.AGGREGATE_TYPE
        self.aggregate_list = cfg.MODEL.AGGREGATE_LIST
        self.data_stamp_dim = 0  # Data stamp is an aggregation of multiple features
        for agg_type in self.aggregate_list:
            if agg_type == "SEMANTIC":
                self.data_stamp_dim += self.semantic_feat_dim
            elif agg_type == "POSE":
                self.data_stamp_dim += self.pose_dim
            else:
                raise NotImplementedError

        # ablation module
        self.recon_data_stamp = cfg.MODEL.RECON_DATA_STAMP
        self.recon_semantic_feature = cfg.MODEL.RECON_SEMANTIC_FEATURE
        self.recon_pose = cfg.MODEL.RECON_POSE
        self.cond_geometry_feature = cfg.MODEL.COND_GEOMETRY_FEATURE
        self.cond_semantic_feature = cfg.MODEL.COND_SEMANTIC_FEATURE
        self.guide_data_consistency = cfg.MODEL.GUIDE_DATA_CONSISTENCY
        self.guide_semantic_consistency = cfg.MODEL.GUIDE_SEMANTIC_CONSISTENCY
        # vision encoder
        # self.vision_encoder = vision_encoder
        # scheduler
        self.num_diffusion_iters = cfg.MODEL.NUM_DIFFUSION_ITERS
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=False,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )
        # noise net
        self.noise_pred_net = noise_pred_net
        self.nets = nn.ModuleDict(
            {
                "noise_pred_net": self.noise_pred_net,
                "positional_embedding": torch_utils.SinusoidalEmbedding(size=self.time_emb_dim),
            }
        ).to(self.device)
        # build embedding key
        # self.build_embedding_key()
        # semantic feature related
        self.semantic_feat_type = cfg.MODEL.SEMANTIC_FEAT_TYPE
        if self.semantic_feat_type == "clip":
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
            self.clip_model.eval()
            self.clip_model.requires_grad_(False)

    def train(self, num_epochs: int, data_loader):
        # Exponential Moving Average
        # accelerates training and improves stability
        # holds a copy of the model weights
        ema = EMAModel(parameters=self.nets.parameters(), power=0.75)

        # Standard ADAM optimizer
        # Note that EMA parametesr are not optimized
        optimizer = torch.optim.AdamW(params=self.nets.parameters(), lr=1e-4, weight_decay=1e-6)

        # Cosine LR schedule with linear warmup
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=len(data_loader) * num_epochs,
        )
        global_step = 0
        best_loss = float("inf")
        best_state_dict = None
        best_epoch  = None
        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(data_loader)
            progress_bar.set_description(f"Epoch {epoch}")
            for it, dt in enumerate(data_loader):
                target = dt["shifted"]["target"].to(self.device).to(torch.float32)
                fixed = dt["shifted"]["fixed"].to(self.device).to(torch.float32)
                pose9d = dt["shifted"]["9dpose"].to(self.device).to(torch.float32)
                # set type of pose9d to float32
                pose9d = pose9d.to(torch.float32)
                B, M = dt["shifted"]["target"].shape[0], 1
                # sample noise to add to actions
                noise = torch.randn((pose9d.shape[0], pose9d.shape[1]), device=self.device)
                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (B*M,),
                    device=self.device,
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = self.noise_scheduler.add_noise(pose9d, noise, timesteps)

                # predict the noise residual
                noise_pred = self.noise_pred_net(
                    noisy_actions, timesteps, global_cond1=target, global_cond2=fixed
                )

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(self.nets.parameters())

                progress_bar.update(1)
                epoch_loss += loss.detach().cpu().item()
                logs = {"loss": epoch_loss/(it+1), "step": global_step}
                progress_bar.set_postfix(**logs)
                global_step += 1
            
            epoch_loss /= len(data_loader)
            # print(f"Epoch {epoch}, loss {epoch_loss}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                best_state_dict = self.nets.state_dict()
            progress_bar.close()
        # copy ema back to model
        ema.copy_to(self.nets.parameters())
        return best_state_dict, best_epoch

    def inference(self, obs, stats: dict, batch_size: int, init_noise: np.ndarray = None, draw_sample: bool = False):
        """Inference with the model"""
        B = batch_size  # inference batch size is 1
        M = obs["sem_feat"].shape[0]  # (M, D)

        sem_feat = obs["sem_feat"].reshape((M, self.semantic_feat_dim))
        geo_feat = obs["geo_feat"].reshape((M, self.geometry_feat_dim))

        # normalize observation
        nsem_feat = normalize_data(sem_feat, stats=stats["sem_feat"])
        # ngeo_feat = normalize_data(geo_feat, stats=stats["geo_feat"])
        nsem_feat = sem_feat
        ngeo_feat = geo_feat

        # device transfer
        nsem_feat = torch.from_numpy(nsem_feat).to(self.device).to(torch.float32).tile((B, 1, 1))
        ngeo_feat = torch.from_numpy(ngeo_feat).to(self.device).to(torch.float32).tile((B, 1, 1))
        nbatch = {
            "sem_feat": nsem_feat,
            "geo_feat": ngeo_feat,
        }

        # infer action
        with torch.no_grad():
            ncond = self.assemble_cond(nbatch, B, M)  # (B*V, pred_horizon, cond_dim)

            if init_noise is None:
                # initialize action from Guassian noise
                noisy_action = torch.randn((B * M, self.action_dim), device=self.device)
                naction = noisy_action
            else:
                # initialize action from given noise
                naction = torch.from_numpy(init_noise).to(self.device).to(torch.float32)

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.nets["noise_pred_net"](
                    sample=naction, timestep=k, global_cond=ncond
                )
                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample
                # gradient guided time consistency
                naction_grad = torch.zeros_like(naction)
                offset = 0
                gamma = 0.1
                if self.guide_data_consistency and self.recon_data_stamp:
                    ndata_stamp_mean = naction[..., offset:offset+self.time_emb_dim].reshape((B, M, self.time_emb_dim)).mean(
                        dim=1, keepdim=True).repeat(1, M, 1).flatten(start_dim=0, end_dim=1)
                    ndata_stamp_grad = naction[..., offset:offset+self.time_emb_dim] - ndata_stamp_mean
                    naction_grad[..., offset:offset+self.time_emb_dim] = gamma * ndata_stamp_grad
                    offset += self.time_emb_dim
                # apply gradient
                naction = naction - naction_grad
                if draw_sample:
                    # DEBUG
                    if k % 10 == 0:
                        naction_cpu = naction.detach().to("cpu").numpy().reshape((B * M, self.action_dim))
                        compare_distribution(naction_cpu, init_noise, dim_end=4, title=f"Time step {k}")

        # unnormalize action
        naction = naction.detach().to("cpu").numpy().reshape((B, M, self.action_dim))
        return self.parse_action(naction, stats)

    def debug_train(self, data_loader):
        """Debug training process"""
        # batch loop
        with tqdm(data_loader, desc="Batch", leave=False) as tepoch:
            for nbatch in tepoch:
                loss_wrt_timestep = list()
                for idx_time_step in range(self.num_diffusion_iters):
                    B = nbatch["sem_feat"].shape[0]  # (B, M, D)
                    M = nbatch["sem_feat"].shape[1]
                    ncond = self.assemble_cond(nbatch, B, M)  # (B*M, cond_dim)
                    naction = self.assemble_action(nbatch, B, M).to(self.device)  # (B*V, pred_horizon, action_dim)
                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=self.device)
                    # sample a diffusion iteration for each data point
                    timesteps = torch.tensor([idx_time_step], device=self.device).tile((B*M,)).long()
                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = self.noise_pred_net(
                        noisy_actions, timesteps, global_cond=ncond
                    )

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # logging
                    loss_cpu = loss.item()
                    loss_wrt_timestep.append(loss_cpu)
                    #
                    noisy_norm = torch.norm(noisy_actions - naction, dim=-1).mean().item()
                    print(f"Time step {idx_time_step}, loss {loss_cpu}, noisy norm {noisy_norm}")
                # Draw loss wrt time step
                plt.plot(loss_wrt_timestep)
                plt.show()

    def debug_inference(self, dataset, sample_size: int = 750, consider_only_one_pair: bool = False, debug: bool = False, shuffle: bool = False, save_path: str = None, save_fig: bool = False, visualize: bool = True, random_index: int = None):
        if sample_size == -1:
            sample_size = len(dataset)
        if consider_only_one_pair:
            sample_size = 300
        eval_data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=sample_size,
            shuffle=shuffle,
            # num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            # pin_memory=True,
            # persistent_workers=True,
        )
        full_data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=shuffle,
            # num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            # pin_memory=True,
            # persistent_workers=True,
        )
        pose9d_full = None
        for _, dt2 in enumerate(full_data_loader):
            pose9d_f = dt2["shifted"]["9dpose"]
            if not consider_only_one_pair:
                pose9d_full = pose9d_f
            break
        for _, dt in enumerate(eval_data_loader):
            target = dt["shifted"]["target"].to(self.device).to(torch.float32)
            fixed = dt["shifted"]["fixed"].to(self.device).to(torch.float32)
            pose9d = dt["shifted"]["9dpose"].to(self.device).to(torch.float32)
            pose9d = pose9d.to(torch.float32)
            transform = dt["shifted"]["transform"].to(torch.float32)
            if consider_only_one_pair:
                if random_index is not None and random_index == 0:
                    random_index = np.random.randint(0, pose9d.shape[0])
                # random_index = 1374
                print(f"Random index: {random_index}")
                save_path = save_path + f"_r{random_index}"
                pose9d_full = pose9d[random_index].unsqueeze(0).repeat(sample_size, 1).detach().cpu().numpy()
            B, _ = target.shape[0], 1
            # sample noise to add to actions
            noise_sample = torch.randn((pose9d.shape[0], pose9d.shape[1]), device="cuda")
            if consider_only_one_pair:
                target_random = target[random_index].unsqueeze(0).repeat(sample_size, 1, 1)
                fixed_random = fixed[random_index].unsqueeze(0).repeat(sample_size, 1, 1)

            with torch.no_grad():
                self.nets.eval()
                timesteps = list(range(len(self.noise_scheduler)))[::-1]
                for _, t in enumerate(tqdm(timesteps, desc="Denoising steps")):
                    t = torch.from_numpy(np.repeat(t, noise_sample.shape[0])).long().to("cuda")  # num_samples is config.eval.batch_size
                    if consider_only_one_pair:
                        residual = self.nets.noise_pred_net(noise_sample, t, target_random, fixed_random)
                    else:
                        residual = self.nets.noise_pred_net(noise_sample, t, target, fixed)
                    noise_sample = self.noise_scheduler.step(residual, t[0], noise_sample).prev_sample
            
            # for i, pred_transform9d in enumerate(noise_sample):
                # pred_transform9d = torch.clamp(pred_transform9d, min=-1.0, max=1.0)
            pred_transform9d = np.mean(noise_sample.detach().cpu().numpy(), axis=0)
            trans = pred_transform9d[:3].T
            v1 = pred_transform9d[3:6].T
            v2 = pred_transform9d[6:].T
            v1 = v1 / np.linalg.norm(v1)
            v2_orthogonal = v2 - np.dot(v2, v1) * v1
            v2 = v2_orthogonal / np.linalg.norm(v2_orthogonal)
            v3 = np.cross(v1, v2)
            rotation = np.column_stack((v1, v2, v3))
            pred_transform_matrix = np.eye(4)
            pred_transform_matrix[:3, :3] = rotation
            pred_transform_matrix[:3, 3] = trans 

            if debug:
                if consider_only_one_pair:
                    visualize_pcd_with_open3d(target_random[random_index].detach().cpu().numpy(), fixed_random[random_index].detach().cpu().numpy(), pred_transform_matrix)
                    pass
                # else:
                #     visualize_pcd_with_open3d(target[0].detach().cpu().numpy(), fixed[0].detach().cpu().numpy(), pred_transform_matrix)
            compare_distribution(pose9d_full, noise_sample.detach().cpu().numpy(), dim_start=0, dim_end=9, title="Pose", save_path=save_path+".png", save_fig=save_fig, visualize=visualize)        
            break
    def save(self, export_path):
        """Save model weights"""
        torch.save(self.nets.state_dict(), export_path)

    def load(self, export_path):
        """Load model weights"""
        state_dict = torch.load(export_path, map_location=self.device)
        self.nets.load_state_dict(state_dict)
        print("Pretrained weights loaded.")

    def assemble_action(self, nbatch, B, M):
        """Dynamically assemble action"""
        action_list = []
        if self.recon_data_stamp:
            data_stamp_list = []
            for aggregate_type in self.aggregate_list:
                if aggregate_type == "SEMANTIC":
                    nsem_feat = nbatch["sem_feat"].to(self.device)
                    data_stamp_list.append(nsem_feat)
                elif aggregate_type == "POSE":
                    npose = nbatch["pose"].to(self.device)
                    data_stamp_list.append(npose)
                else:
                    raise NotImplementedError
            ndata_stamp = torch.cat(data_stamp_list, dim=-1)  # (B, M, D)
            if self.aggregate_type == "mean":
                ndata_stamp = ndata_stamp.mean(dim=1, keepdim=True).repeat(1, M, 1).flatten(start_dim=0, end_dim=1)
            elif self.aggregate_type == "sum":
                ndata_stamp = ndata_stamp.sum(dim=1, keepdim=True).repeat(1, M, 1).flatten(start_dim=0, end_dim=1)
            elif self.aggregate_type == "max":
                ndata_stamp = ndata_stamp.max(dim=1, keepdim=True).repeat(1, M, 1).flatten(start_dim=0, end_dim=1)
            else:
                raise NotImplementedError
            action_list.append(ndata_stamp)
        if self.recon_semantic_feature:
            nsem_feat = nbatch["sem_feat"].to(self.device)  # (B, M, D)
            nsem_feat = nsem_feat.flatten(start_dim=0, end_dim=1)  # (B * M, D)
            action_list.append(nsem_feat)
        if self.recon_pose:
            npose = nbatch["pose"].to(self.device)
            npose = npose.flatten(start_dim=0, end_dim=1)  # (B * M, D)
            action_list.append(npose)
        if len(action_list) == 0:
            naction = None
        else:
            naction = torch.cat(action_list, dim=-1)  # (B * M, action_dim)
        return naction

    def assemble_cond(self, nbatch, B, M):
        """Dynamically assemble condition"""
        cond_list = []
        if self.cond_semantic_feature:
            nsem_feat = nbatch["sem_feat"].to(self.device)  # (B, M, D)
            nsem_feat = nsem_feat.flatten(start_dim=0, end_dim=1)  # (B * M, D)
            cond_list.append(nsem_feat)
        if self.cond_geometry_feature:
            ngeo_feat = nbatch["geo_feat"].to(self.device)  # (B, M, D)
            ngeo_feat = ngeo_feat.flatten(start_dim=0, end_dim=1)  # (B * M, D)
            cond_list.append(ngeo_feat)
        if len(cond_list) == 0:
            ncond = None
        else:
            ncond = torch.cat(cond_list, dim=-1)  # (B * M, pred_horizon, cond_dim)
        return ncond

    def parse_action(self, naction, stats):
        """Parse action into timestamp and pose"""
        offset = 0
        actions = {}
        if self.recon_data_stamp:
            ndata_stamp = naction[..., offset:offset + self.data_stamp_dim]
            actions["data_stamp"] = ndata_stamp
            offset += self.data_stamp_dim
        if self.recon_semantic_feature:
            nsem_feat = naction[..., offset:offset + self.semantic_feat_dim]
            actions["sem_feat"] = unnormalize_data(nsem_feat, stats=stats["sem_feat"])
            offset += self.semantic_feat_dim
        if self.recon_pose:
            npose = naction[..., offset:offset + self.pose_dim]
            actions["pose"] = unnormalize_data(npose, stats=stats["pose"])
            offset += self.pose_dim
        return actions

    def build_embedding_key(self, max_len=1000):
        """Build embedding key"""
        self.embedding_key = torch.arange(max_len, device=self.device).reshape((-1))
        self.embedding_key = self.nets["positional_embedding"](self.embedding_key).detach().to("cpu").numpy()
        self.embedding_key = self.embedding_key / np.linalg.norm(self.embedding_key, axis=-1, keepdims=True)  # (max_len, time_emb_dim)
        compare_distribution(self.embedding_key, None, dim_end=8)

    def parse_embedding(self, embedding_query):
        """Parse embedding query into timestamp and pose"""
        B, M, D = embedding_query.shape
        embedding_query = embedding_query.reshape((-1, self.time_emb_dim))
        # normalize
        embedding_query = embedding_query / np.linalg.norm(embedding_query, axis=-1, keepdims=True)
        embedding_attention = np.dot(embedding_query, self.embedding_key.T)
        # argmax along the last dimension
        embedding_attention = np.argmax(embedding_attention, axis=-1)
        return embedding_attention.reshape((B, M, 1))

    def encode_text(self, text: str, idx: int):
        """Encode text into semantic feature"""
        if self.semantic_feat_type == "clip":
            with torch.no_grad():
                text_inputs = torch.cat([clip.tokenize(text)]).to(self.device)
                semantic_feature = self.clip_model.encode_text(text_inputs).detach().cpu().numpy()[0]
            return semantic_feature
        elif self.semantic_feat_type == "one_hot":
            semantic_feature = np.zeros((self.semantic_feat_dim,), dtype=np.float32)
            semantic_feature[idx] = 1.0
            return semantic_feature
        else:
            raise NotImplementedError

    def parse_vocab(self, sem_feats: np.ndarray, vocab: list[str]):
        """Parse semantic feature to vocabulary"""
        # Build vocabulary dictionary
        B, M, D = sem_feats.shape
        sem_feats = sem_feats.reshape((-1, D))
        vocab_dict = {}
        for i, obj_name in enumerate(vocab):
            vocab_dict[obj_name] = self.encode_text(obj_name, i)
        vocab_feats = np.stack(list(vocab_dict.values()))
        # Match semantic feature to vocabulary; by nearest neighbor
        vocab_names = [[]]
        vocab_ids = [[]]
        idx_batch = 0
        for i in range(sem_feats.shape[0]):
            if i % M == 0 and i > 0:
                idx_batch += 1
                vocab_names.append([])
                vocab_ids.append([])
            # nsem_feat = sem_feats[i] / np.linalg.norm(sem_feats[i])
            dist = np.linalg.norm(sem_feats[i] - vocab_feats, axis=-1)
            vocab_names[idx_batch].append(vocab[np.argmin(dist)])
            vocab_ids[idx_batch].append(np.argmin(dist) + 1)  # vocab id starts from 1
        return vocab_names, vocab_ids
