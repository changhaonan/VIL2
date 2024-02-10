import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PcdPairCollator:
    def __call__(self, samples):
        target = {
            "target_coord": [],
            "target_feat": [],
            "fixed_coord": [],
            "fixed_feat": [],
            "target_pose": [],
            "target_batch_index": [],
            "fixed_batch_index": [],
            "is_valid_crop": [],
        }
        for sample_id, item in enumerate(samples):
            target["target_coord"].append(item["target_coord"])  # (N, 3)
            target["target_feat"].append(item["target_feat"])  # (N, 3)
            target["target_batch_index"].append(np.full([len(item["target_coord"])], fill_value=sample_id))  # (N,)
            target["fixed_coord"].append(item["fixed_coord"])  # (M, 3)
            target["fixed_feat"].append(item["fixed_feat"])  # (M, 3)
            target["fixed_batch_index"].append(np.full([len(item["fixed_coord"])], fill_value=sample_id))  # (M,)
            target["target_pose"].append(item["target_pose"][None, :])  #
            target["is_valid_crop"].append(item["is_valid_crop"])

        return {k: torch.from_numpy(np.concatenate(v)) for k, v in target.items()}


if __name__ == "__main__":
    import os
    from vil2.data.pcd_dataset import PcdPairDataset

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
        noise_level=0.5,
        crop_pcd=False,
        crop_size=0.4,
    )
    dataset.set_mode("train")

    # Test data loader
    collate_fn = PcdPairCollator()
    for d in DataLoader(dataset, collate_fn=collate_fn, batch_size=4):
        print(d["target_batch_index"].shape)
        print(d["fixed_batch_index"].shape)
        print(d["target_coord"].shape)
        print(d["target_feat"].shape)
        print(d["fixed_coord"].shape)
        print(d["fixed_feat"].shape)
        print(d["target_pose"].shape)
        break
