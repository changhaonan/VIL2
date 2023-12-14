from vil2.utils.eval_utils import compare_distribution
from vil2.data_gen.data_loader import visualize_pcd_with_open3d
import torch 
import os 
import pickle
# from vil2.data_gen.data_loader import DiffDataset

root_path = os.path.dirname((os.path.abspath(__file__)))

# with open(os.path.join(root_path, "test_data", "dmorp_augmented", "diffusion_dataset.pkl"), "rb") as f:
#     dtset = pickle.load(f)
# dataset = DiffDataset(dtset=dtset)
# full_data_loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=len(dataset),
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True,
#     persistent_workers=True,
# )
# pose9d_full = None
# for _, dt in enumerate(full_data_loader):
#     pose9d_full = dt["shifted"]["9dpose"].detach().cpu().numpy()
#     compare_distribution(pose9d_full, None, dim_start=0, dim_end=9, title="Pose") 
#     break

dtset = [
    {
        "original" :
                        {
                            "a" : 1,
                            "b" : 2,
                            "c" : 3
                        },
        "shifted" :
                        {
                            "a" : -1,
                            "b" : -2,
                            "c" : -3
                        }
    },
    {
        "original" :
                        {
                            "a" : 4,
                            "b" : 5,
                            "c" : 6
                        },
        "shifted" :
                        {
                            "a" : -4,
                            "b" : -5,
                            "c" : -6
                        }
    },
    {
        "original" :
                        {
                            "a" : 7,
                            "b" : 8,
                            "c" : 9
                        },
        "shifted" :
                        {
                            "a" : -7,
                            "b" : -8,
                            "c" : -9
                        }
    },
    {
        "original" :
                        {
                            "a" : 10,
                            "b" : 11,
                            "c" : 12
                        },
        "shifted" :
                        {
                            "a" : -10,
                            "b" : -11,
                            "c" : -12
                        }
    }
]

full_data_loader = torch.utils.data.DataLoader(
    dtset,
    batch_size=len(dtset),
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

for _, dt in enumerate(full_data_loader):
    print(dt)
    print("====================\n")