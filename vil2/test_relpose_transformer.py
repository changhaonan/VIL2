"""Test pose transformer."""

import os
import open3d as o3d
import torch
import numpy as np
import pickle
import argparse
import vil2.utils.misc_utils as utils
from vil2.data.pcd_dataset import PcdPairDataset
from vil2.data.pcd_datalodaer import PcdPairCollator
from vil2.model.network.relpose_transformer import RelPoseTransformer
from vil2.model.rpt_model import RPTModel
from detectron2.config import LazyConfig
from torch.utils.data.dataset import random_split
from vil2.vil2_utils import build_dmorp_dataset
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
    argparser.add_argument(
        "--task_name",
        type=str,
        default="book_in_bookshelf",
        help="stack_can_in_cabinet, book_in_bookshelf, mug_on_rack_multi",
    )
    args = argparser.parse_args()
    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    task_name = args.task_name
    root_path = os.path.dirname((os.path.abspath(__file__)))
    cfg_file = os.path.join(root_path, "config", f"pose_transformer_rpdiff_{task_name}.py")
    cfg = LazyConfig.load(cfg_file)

    # Use raw data
    # Prepare path
    data_path_dict = {
        "stack_can_in_cabinet": "/home/harvey/Project/VIL2/vil2/external/rpdiff/data/task_demos/can_in_cabinet_stack/task_name_stack_can_in_cabinet",
        "book_in_bookshelf": "/home/harvey/Project/VIL2/vil2/external/rpdiff/data/task_demos/book_on_bookshelf_double_view_rnd_ori/task_name_book_in_bookshelf",
        "mug_on_rack_multi": "/home/harvey/Project/VIL2/vil2/external/rpdiff/data/task_demos/mug_on_rack_multi_large_proc_gen_demos/task_name_mug_on_rack_multi",
    }
    data_format = "raw"  # "rpdiff_fail" or "raw", "test"
    rpdiff_path = data_path_dict[task_name]
    rpdiff_file_list = os.listdir(rpdiff_path)
    rpdiff_file_list = [f for f in rpdiff_file_list if f.endswith(".npz")]

    # Load dataset & data loader
    train_dataset, val_dataset, test_dataset = build_dmorp_dataset(root_path, cfg)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=PcdPairCollator(),
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=PcdPairCollator(),
    )

    # Test dataset
    if data_format == "rpdiff_fail":
        failed_data_path = "/home/harvey/Project/VIL2/vil2/external/rpdiff/eval_data/eval_data/can_on_cabinet_nosc/seed_0/failed"
        failed_data_list = os.listdir(failed_data_path)
        failed_data_list = [os.path.join(failed_data_path, f) for f in failed_data_list if f.endswith(".npz")]

    # Build model
    net_name = cfg.MODEL.NOISE_NET.NAME
    net_init_args = cfg.MODEL.NOISE_NET.INIT_ARGS[net_name]
    pose_transformer = RelPoseTransformer(**net_init_args)
    rpt_model = RPTModel(cfg, pose_transformer)

    model_name = rpt_model.experiment_name()
    noise_net_name = cfg.MODEL.NOISE_NET.NAME
    save_dir = os.path.join(root_path, "test_data", task_name, "checkpoints", noise_net_name)
    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = f"{save_path}/checkpoints"
    # Select the best checkpoint
    checkpoints = os.listdir(checkpoint_path)
    sorted_checkpoints = sorted(checkpoints, key=lambda x: float(x.split("=")[-1].split(".ckpt")[0]))
    checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])

    rpt_model.load(checkpoint_file)
    for i in range(20):
        i = 2
        if data_format == "test":
            sample_batch = next(iter(test_data_loader))
        elif data_format == "raw":
            raw_data = np.load(os.path.join(rpdiff_path, rpdiff_file_list[i]), allow_pickle=True)
            anchor_coord_s, target_coord_s = parse_child_parent(raw_data["multi_obj_start_pcd"])
            anchor_pose_s, target_pose_s = parse_child_parent(raw_data["multi_obj_start_obj_pose"])
            anchor_pose_f, target_pose_f = parse_child_parent(raw_data["multi_obj_final_obj_pose"])
            data = rpt_model.preprocess_input_rpdiff(
                anchor_coord=anchor_coord_s,
                target_coord=target_coord_s,
            )
        elif data_format == "rpdiff_fail":
            failed_data_file = failed_data_list[i]
            failed_data = np.load(failed_data_file, allow_pickle=True)
            anchor_coord_s = failed_data["parent_pcd"]
            target_coord_s = failed_data["child_pcd"]
            final_child_pcd = failed_data["final_child_pcd"]
            data = rpt_model.preprocess_input_rpdiff(
                anchor_coord=anchor_coord_s,
                target_coord=target_coord_s,
            )

        target_coord = data["target_coord"]
        target_feat = data["target_feat"]
        anchor_coord = data["anchor_coord"]
        anchor_feat = data["anchor_feat"]

        # Init crop center
        crop_size = cfg.DATALOADER.AUGMENTATION.CROP_SIZE
        knn_k = cfg.DATALOADER.AUGMENTATION.KNN_K
        crop_strategy = cfg.DATALOADER.AUGMENTATION.CROP_STRATEGY
        num_grid = cfg.MODEL.NUM_GRID
        sample_size = cfg.MODEL.SAMPLE_SIZE
        sample_strategy = cfg.MODEL.SAMPLE_STRATEGY
        # Batch sampling
        sample_batch, samples = rpt_model.batch_random_sample(
            sample_size,
            sample_strategy=sample_strategy,
            target_coord=target_coord,
            target_feat=target_feat,
            anchor_coord=anchor_coord,
            anchor_feat=anchor_feat,
            crop_strategy=crop_strategy,
            crop_size=crop_size,
            knn_k=knn_k,
            num_grid=num_grid,
        )

        pred_pose9d, pred_status = rpt_model.predict(batch=sample_batch)

        # Rank the prediction by status
        sorted_indices = np.argsort(pred_status[:, 1])
        sorted_indices = sorted_indices[::-1]
        pred_pose9d = pred_pose9d[sorted_indices]
        pred_status = pred_status[sorted_indices]

        # Check the result
        anchor_pcd = o3d.geometry.PointCloud()
        anchor_pcd.points = o3d.utility.Vector3dVector(anchor_coord)
        anchor_pcd.paint_uniform_color([0, 1, 0])

        # Check crop sampling
        for j in range(sample_size):
            print(f"Status: {pred_status[j]} for {j}-th sample")
            vis_list = [anchor_pcd]
            # Crop fixed
            crop_anchor_coord = samples[sorted_indices[j]]["anchor_coord"]
            crop_anchor_pcd = o3d.geometry.PointCloud()
            crop_anchor_pcd.points = o3d.utility.Vector3dVector(crop_anchor_coord)
            crop_anchor_pcd.paint_uniform_color([1, 0, 0])
            crop_center = samples[sorted_indices[j]]["crop_center"]
            crop_anchor_pcd.translate(crop_center)
            vis_list.append(crop_anchor_pcd)

            # Target
            pred_pose_mat = utils.pose9d_to_mat(pred_pose9d[j], rot_axis=cfg.DATALOADER.AUGMENTATION.ROT_AXIS)
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_coord)
            target_pcd.paint_uniform_color([0, 0, 1])
            target_pcd.transform(pred_pose_mat)
            target_pcd.translate(crop_center)
            vis_list.append(target_pcd)

            crop_center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            crop_center_sphere.translate(crop_center)
            vis_list.append(crop_center_sphere)

            # Origin
            origin_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            vis_list.append(origin_pcd)
            o3d.visualization.draw_geometries(vis_list)

        for j in range(3):
            print(f"Status: {pred_status[j]} for {j}-th sample")

            # DEBUG: check the recovered pose
            recover_pose = rpt_model.pose_recover_rpdiff(pred_pose9d[j], samples[sorted_indices[j]]["crop_center"], data)
            # DEBUG:
            anchor_coord_o3d = o3d.geometry.PointCloud()
            anchor_coord_o3d.points = o3d.utility.Vector3dVector(anchor_coord_s)
            # anchor_coord_o3d.paint_uniform_color([0, 1, 0])
            target_coord_o3d = o3d.geometry.PointCloud()
            target_coord_o3d.points = o3d.utility.Vector3dVector(target_coord_s)
            target_coord_o3d.paint_uniform_color([1, 1, 0])
            target_coord_o3d.transform(recover_pose)

            origin_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([origin_pcd, target_coord_o3d, anchor_coord_o3d])
