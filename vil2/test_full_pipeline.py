"""Test full pipeline, including pcd segmentation and rel pose transformer."""

import os
import open3d as o3d
import torch
import numpy as np
import pickle
import argparse
import vil2.utils.misc_utils as utils
from vil2.data.pcd_dataset import PcdPairDataset
from vil2.data.pcd_datalodaer import PcdPairCollator
from vil2.model.network.rigpose_transformer import RigPoseTransformer
from vil2.model.network.pcd_seg_noise_net import PcdSegNoiseNet
from vil2.model.pcdd_seg_model import PCDDModel
from vil2.model.rgt_model import RGTModel
from detectron2.config import LazyConfig
from torch.utils.data.dataset import random_split
from vil2.vil2_utils import build_dmorp_dataset
from vil2.utils.rpdiff_utils import preprocess_input_rpdiff
import random


# Read data
def parse_child_parent(arr):
    pcd_dict = arr[()]
    parent_val = pcd_dict["parent"]
    child_val = pcd_dict["child"]
    return parent_val, child_val


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--random_index", type=int, default=0)
    argparser.add_argument("--task_name", type=str, default="book_in_bookshelf", help="stack_can_in_cabinet, book_in_bookshelf, mug_on_rack_multi")
    args = argparser.parse_args()
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load config
    task_name = args.task_name
    root_path = os.path.dirname((os.path.abspath(__file__)))
    cfg_file = os.path.join(root_path, "config", f"pose_transformer_rpdiff_{task_name}.py")
    act_cfg = LazyConfig.load(cfg_file)
    seg_cfg = LazyConfig.load(cfg_file)
    # Overriding config
    act_cfg.MODEL.NOISE_NET.NAME = "RGTModel"
    act_cfg.DATALOADER.AUGMENTATION.CROP_PCD = True
    act_cfg.DATALOADER.BATCH_SIZE = 32
    seg_cfg.MODEL.NOISE_NET.NAME = "PCDSAMNOISENET"
    seg_cfg.DATALOADER.AUGMENTATION.CROP_PCD = False
    seg_cfg.DATALOADER.BATCH_SIZE = 8

    # Load dataset & data loader
    train_dataset, val_dataset, test_dataset = build_dmorp_dataset(root_path, seg_cfg)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=seg_cfg.DATALOADER.BATCH_SIZE, shuffle=True, num_workers=act_cfg.DATALOADER.NUM_WORKERS, collate_fn=PcdPairCollator())
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=seg_cfg.DATALOADER.BATCH_SIZE, shuffle=False, num_workers=act_cfg.DATALOADER.NUM_WORKERS, collate_fn=PcdPairCollator())

    # Build segmentation model
    net_name = seg_cfg.MODEL.NOISE_NET.NAME
    net_init_args = seg_cfg.MODEL.NOISE_NET.INIT_ARGS[net_name]
    seg_net = PcdSegNoiseNet(**net_init_args)
    seg_model = PCDDModel(seg_cfg, seg_net)
    model_name = seg_model.experiment_name()
    seg_net_name = seg_cfg.MODEL.NOISE_NET.NAME
    save_dir = os.path.join(root_path, "test_data", task_name, "checkpoints", seg_net_name)
    save_path = os.path.join(save_dir, model_name)
    checkpoint_path = f"{save_path}/checkpoints"
    checkpoints = os.listdir(checkpoint_path)
    sorted_checkpoints = sorted(checkpoints, key=lambda x: float(x.split("=")[-1].split(".ckpt")[0]))
    checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])
    seg_model.load(checkpoint_file)

    # Build action model
    act_net_name = act_cfg.MODEL.NOISE_NET.NAME
    net_init_args = act_cfg.MODEL.NOISE_NET.INIT_ARGS[act_net_name]
    act_net = RigPoseTransformer(**net_init_args)
    act_model = RGTModel(act_cfg, act_net)
    seg_model_name = act_model.experiment_name()
    seg_net_name = act_cfg.MODEL.NOISE_NET.NAME
    save_dir = os.path.join(root_path, "test_data", task_name, "checkpoints", seg_net_name)
    save_path = os.path.join(save_dir, seg_model_name)
    checkpoint_path = f"{save_path}/checkpoints"
    checkpoints = os.listdir(checkpoint_path)
    sorted_checkpoints = sorted(checkpoints, key=lambda x: float(x.split("=")[-1].split(".ckpt")[0]))
    checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])
    act_model.load(checkpoint_file)

    # Testing raw material
    for i in range(20):
        batch = next(iter(val_data_loader))
        if i < 11:
            continue
        # Perform segmentation
        check_batch_idx = 1
        pred_anchor_label, anchor_coord, anchor_normal, anchor_feat = seg_model.predict(batch=batch, check_batch_idx=check_batch_idx, vis=False)
        seg_list = seg_model.seg_and_rank(anchor_coord, pred_anchor_label, normal=anchor_normal, feat=anchor_feat, crop_strategy="bbox")

        # Visualize
        # for seg in seg_list:
        #     print(f"Seg prob: {seg['prob']}")
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(seg["coord"])
        #     o3d.visualization.draw_geometries([pcd])

        # Prepare data
        target_batch_idx = batch["target_batch_index"]
        target_coord = batch["target_coord"][target_batch_idx == check_batch_idx].numpy()
        target_normal = batch["target_normal"][target_batch_idx == check_batch_idx].numpy()
        target_feat = batch["target_feat"][target_batch_idx == check_batch_idx].numpy()
        # Cropped coord
        anchor_coord = seg_list[0]["coord"]
        anchor_normal = seg_list[0]["normal"]
        anchor_feat = seg_list[0]["feat"]
        # Full coord
        anchor_batch_idx = batch["anchor_batch_index"]
        anchor_coord_full = batch["anchor_coord"][anchor_batch_idx == check_batch_idx].numpy()
        anchor_normal_full = batch["anchor_normal"][anchor_batch_idx == check_batch_idx].numpy()
        anchor_feat_full = batch["anchor_feat"][anchor_batch_idx == check_batch_idx].numpy()

        # Move anchor to center
        anchor_center = np.mean(anchor_coord, axis=0)
        anchor_coord -= anchor_center
        for k in range(1):
            print(f"Batch {i}, Iteration {k}...")
            if k != 0:
                batch["prev_R"] = torch.from_numpy(pred_R)
                batch["prev_t"] = torch.from_numpy(pred_t)
            conf_matrix, gt_corr, (pred_R, pred_t) = act_model.predict(
                target_coord=target_coord, target_normal=target_normal, target_feat=target_feat, anchor_coord=anchor_coord, anchor_normal=anchor_normal, anchor_feat=anchor_feat, vis=True
            )
        # Visualize everything together
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_coord)
        target_pcd.normals = o3d.utility.Vector3dVector(target_normal)
        target_pcd.paint_uniform_color([1, 0, 0])
        pose = np.eye(4)
        pose[:3, :3] = pred_R
        pose[:3, 3] = pred_t
        target_pcd.transform(pose)
        target_pcd.translate(anchor_center)

        anchor_pcd = o3d.geometry.PointCloud()
        anchor_pcd.points = o3d.utility.Vector3dVector(anchor_coord_full)
        anchor_pcd.normals = o3d.utility.Vector3dVector(anchor_normal_full)
        anchor_pcd.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([target_pcd, anchor_pcd])
