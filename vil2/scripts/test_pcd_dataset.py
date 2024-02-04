import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PointCloudCollator:
    def __call__(self, samples):
        target = {"pcd": [], "batch_index": []}
        for sample_id, item in enumerate(samples):
            target["pcd"].append(item["pcd"])  # (N, 3)
            target["batch_index"].append(np.full([len(item["pcd"])], fill_value=sample_id))
        return {k: torch.from_numpy(np.concatenate(v)) for k, v in target.items()}


class PointCloudDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, _):
        return {"pcd": np.random.rand(np.random.randint(10, 100), 3)}


if __name__ == "__main__":
    dataset = PointCloudDataset()
    collate_fn = PointCloudCollator()
    for d in DataLoader(dataset, collate_fn=collate_fn, batch_size=4):
        print(d["batch_index"])
        print(d["pcd"].shape)