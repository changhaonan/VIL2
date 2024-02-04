"""Test point cloud transformer"""
import os
from torch.utils.data import Dataset, DataLoader
from vil2.data.pcd_dataset import PcdPairDataset
from vil2.data.pcd_datalodaer import PcdPairCollator
from vil2.model.network.geometric import PointTransformerNetwork, batch2offset


if __name__ == "__main__":
    dataset_name = "dmorp_rdiff"
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split = "test"
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
        crop_pcd=False,
    )
    dataset.set_mode("train")

    # Init network
    pct = PointTransformerNetwork(
        grid_sizes=[0.015, 0.03],
        depths=[2, 3, 3],
        dec_depths=[1, 1],
        hidden_dims=[64, 128, 256],
        n_heads=[4, 8, 8],
        ks=[16, 24, 32],
        in_dim=3
    )


    # Test data loader
    collate_fn = PcdPairCollator()
    for d in DataLoader(dataset, collate_fn=collate_fn, batch_size=4):
        # Assemble input
        coord = d["target_coord"]
        feat = d["target_feat"]
        batch_idx = d["target_batch_index"]
        offset = batch2offset(batch_idx)
        points = [coord, feat, offset]
        points, all_points, cluster_indexes = pct(points, return_full=True)
        pass
    