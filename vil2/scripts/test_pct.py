"""Test point cloud transformer"""

import torch
import os
import random
from torch.utils.data import Dataset, DataLoader
from vil2.data.pcd_dataset import PcdPairDataset
from vil2.data.pcd_datalodaer import PcdPairCollator
from vil2.model.network.geometric import PointTransformerNetwork, batch2offset, to_dense_batch, offset2batch


if __name__ == "__main__":
    # Set seed
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    dataset_name = "dmorp_rdiff"
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split = "test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Test data loader
    dataset = PcdPairDataset(
        data_file_list=[f"{root_dir}/test_data/{dataset_name}/diffusion_dataset_0_2048_s25000-c1-r0.5_{split}.pkl"],
        dataset_name="dmorp",
        add_colors=True,
        add_normals=True,
        is_elastic_distortion=False,
        is_random_distortion=True,
        volume_augmentations_path=f"{root_dir}/config/va_rotation.yaml",
        noise_level=0.1,
        crop_pcd=True,
        crop_size=0.2,
    )
    dataset.set_mode("train")

    # Init network
    pct = PointTransformerNetwork(
        grid_sizes=[0.03, 0.05],
        depths=[2, 3, 3],
        dec_depths=[1, 1],
        hidden_dims=[64, 128, 256],
        n_heads=[4, 8, 8],
        ks=[16, 24, 32],
        in_dim=6,
    ).to(device)

    # Test data loader
    collate_fn = PcdPairCollator()
    for i in range(100):
        for batch in DataLoader(dataset, collate_fn=collate_fn, batch_size=32, shuffle=True):
            # Assemble input
            target_coord = batch["target_coord"].to(device)
            target_feat = batch["target_feat"].to(device)
            target_batch_idx = batch["target_batch_index"].to(device)
            anchor_coord = batch["anchor_coord"].to(device)
            anchor_feat = batch["anchor_feat"].to(device)
            anchor_batch_idx = batch["anchor_batch_index"].to(device)

            target_offset = batch2offset(target_batch_idx)
            anchor_offset = batch2offset(anchor_batch_idx)
            target_points = [target_coord, target_feat, target_offset]
            anchor_points = [anchor_coord, anchor_feat, anchor_offset]

            # Assemble batch
            batch_anchor_feat, mask = to_dense_batch(anchor_feat, anchor_batch_idx)

            # Encode feature
            enc_target_points = pct(target_points)
            enc_anchor_points, all_enc_anchor_points, cluster_indexes = pct(anchor_points, return_full=True)

            enc_target_coord, enc_target_feat, enc_target_offset = enc_target_points
            # Convert to batch & mask
            enc_target_batch_index = offset2batch(enc_target_offset)
            enc_target_feat, tenc_target_feat_mask = to_dense_batch(enc_target_feat, enc_target_batch_index)
            enc_target_feat_mask = torch.cat(
                (torch.ones_like(tenc_target_feat_mask[:, :1]), tenc_target_feat_mask), dim=1
            )

            for enc_anchor_point in all_enc_anchor_points:
                enc_anchor_coord, enc_anchor_feat, enc_anchor_offset = enc_anchor_point
                enc_anchor_batch_index = offset2batch(enc_anchor_offset)
                enc_anchor_feat, enc_anchor_feat_mask = to_dense_batch(enc_anchor_feat, enc_anchor_batch_index)
                enc_anchor_feat_mask = enc_anchor_feat_mask == 0
                print(
                    f"enc_target_feat: {enc_target_feat.shape}, enc_anchor_feat: {enc_anchor_feat.shape}, anchor_feat: {batch_anchor_feat.shape}, enc_target_mask: {enc_target_feat_mask.shape}, enc_anchor_mask: {enc_anchor_feat_mask.shape}"
                )
                if enc_anchor_feat.shape[0] != batch_anchor_feat.shape[0]:
                    print(anchor_offset)
                    print(target_offset)

            # Release memory
            del target_coord, target_feat, target_batch_idx, anchor_coord, anchor_feat, anchor_batch_idx
