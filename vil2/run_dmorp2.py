"""Run Diffusion Model for Object Relative Pose Generation"""
import os
import torch
import pickle
import argparse
import copy
from vil2.data_gen.data_loader import DiffDataset
from vil2.model.dmorp_model import DmorpModel
from vil2.model.net_factory import build_noise_pred_net
from detectron2.config import LazyConfig
from torch.utils.data.dataset import random_split


if __name__ == "__main__":
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--random_index", type=int, default=0)
    args = argparser.parse_args()
    # Load config
    task_name = "Dmorp"
    root_path = os.path.dirname((os.path.abspath(__file__)))
    cfg_file = os.path.join(root_path, "config", "dmorp_simplify.py")
    cfg = LazyConfig.load(cfg_file)
    retrain = cfg.MODEL.RETRAIN
    pcd_size = cfg.MODEL.PCD_SIZE
    # Load dataset & data loader
    with open(os.path.join(root_path, "test_data", "dmorp_augmented", f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}.pkl"), "rb") as f:
        dtset = pickle.load(f)
    dataset = DiffDataset(dtset=dtset)
    train_size = int(cfg.MODEL.TRAIN_TEST_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    if os.path.exists(os.path.join(root_path, "test_data", "dmorp_augmented", f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_train.pkl")):
        print("Loading cached dataset....")
        with open(os.path.join(root_path, "test_data", "dmorp_augmented", f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_train.pkl"), "rb") as f:
            train_dataset = pickle.load(f)
        with open(os.path.join(root_path, "test_data", "dmorp_augmented", f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_val.pkl"), "rb") as f:
            val_dataset = pickle.load(f)
    else:
        print("Caching dataset....")
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        with open(os.path.join(root_path, "test_data", "dmorp_augmented", f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_train.pkl"), "wb") as f:
            pickle.dump(train_dataset, f)
        with open(os.path.join(root_path, "test_data", "dmorp_augmented", f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_val.pkl"), "wb") as f:
            pickle.dump(val_dataset, f)
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=True,
        # num_workers=cfg.DATALOADER.NUM_WORKERS,
        # pin_memory=True,
        # persistent_workers=True,
    )

    # Compute network input/output dimension
    noise_net_name = cfg.MODEL.NOISE_NET.NAME
    noise_net_init_args = cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name]
    input_dim = 0
    global_cond_dim = 0
    # condition related
    cond_geometry_feature = cfg.MODEL.COND_GEOMETRY_FEATURE
    cond_semantic_feature = cfg.MODEL.COND_SEMANTIC_FEATURE
    # i/o related
    recon_data_stamp = cfg.MODEL.RECON_DATA_STAMP
    recon_semantic_feature = cfg.MODEL.RECON_SEMANTIC_FEATURE
    recon_pose = cfg.MODEL.RECON_POSE
    aggregate_list = cfg.MODEL.AGGREGATE_LIST
    if recon_data_stamp:
        for agg_type in aggregate_list:
            if agg_type == "SEMANTIC":
                input_dim += cfg.MODEL.SEMANTIC_FEAT_DIM
            elif agg_type == "POSE":
                input_dim += cfg.MODEL.POSE_DIM
            else:
                raise NotImplementedError
    if recon_semantic_feature:
        input_dim += cfg.MODEL.SEMANTIC_FEAT_DIM
    if recon_pose:
        input_dim += cfg.MODEL.POSE_DIM
    if cond_geometry_feature:
        global_cond_dim += cfg.MODEL.GEOMETRY_FEAT_DIM
    if cond_semantic_feature:
        global_cond_dim += cfg.MODEL.SEMANTIC_FEAT_DIM
    noise_net_init_args["input_dim"] = input_dim
    noise_net_init_args["global_cond_dim"] = global_cond_dim
    noise_net_init_args["diffusion_step_embed_dim"] = cfg.MODEL.TIME_EMB_DIM
    dmorp_model = DmorpModel(
        cfg,
        vision_encoder=None,
        noise_pred_net=build_noise_pred_net(
            noise_net_name, **noise_net_init_args
        ),
        device="cuda:0"
    )
    model_name = f"dmorp_model_rel_n{noise_net_name}"
    model_name += f"_ps{pcd_size}"
    model_name += f"_ndi{cfg.MODEL.NUM_DIFFUSION_ITERS}" 
    model_name += f"_ne{cfg.TRAIN.NUM_EPOCHS}" 
    model_name += f"_bs{cfg.DATALOADER.BATCH_SIZE}"
    model_name += f"_ug{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].use_global_geometry}"
    model_name += f"_id{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].input_dim}"
    model_name += f"_cgf{cfg.MODEL.COND_GEOMETRY_FEATURE}"    
    model_name += f"_dsed{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].diffusion_step_embed_dim}"
    model_name += f"_up{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].use_pointnet}"
    model_name += f"_ro{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].rotation_orthogonalization}"
    if noise_net_name == "UNETMLP":
        model_name += f"_dd{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].down_dims}"
        model_name += f"_dpe{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].downsample_pcd_enc}"
        model_name += f"_ds{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].downsample_size}"
    elif noise_net_name == "TRANSFORMER":
        model_name += f"_nah{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].num_attention_heads}"
        model_name += f"_ehd{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].encoder_hidden_dim}"
        model_name += f"_ed{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].encoder_dropout}"
        model_name += f"_ea{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].encoder_activation}"
        model_name += f"_enl{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].encoder_num_layers}"
        model_name += f"_fpd{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].fusion_projection_dim}"
    elif noise_net_name == "PARALLEL_MLP":
        model_name += f"_fpd{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].fusion_projection_dim}"
    
    model_name += f"_ro{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].rotation_orthogonalization}"
    model_name += f"_uds{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].use_dropout_sampler}"
    model_name += ".pt"

    save_dir = os.path.join(root_path, "test_data", task_name, "checkpoints", noise_net_name)
    os.makedirs(save_dir, exist_ok=True)
    if retrain:
        # save the data
        save_path = os.path.join(save_dir, model_name)
        # dmorp_model.module.train(num_epochs=cfg.TRAIN.NUM_EPOCHS, data_loader=data_loader)
        best_model, best_epoch = dmorp_model.train(num_epochs=cfg.TRAIN.NUM_EPOCHS, data_loader=data_loader, save_path=save_path)
        # best_model, best_epoch = dmorp_model.train_score(num_epochs=cfg.TRAIN.NUM_EPOCHS, data_loader=data_loader)
        torch.save(best_model, save_path)
        print(f"Saving the best epoch:{best_epoch}. Model saved to {save_path}")
    else:
        # load the model
        save_path = os.path.join(save_dir, model_name)
        dmorp_model.nets.load_state_dict(torch.load(save_path))
        print(f"Model loaded from {save_path}")

    # Test inference
    # Load vocabulary
    res_path = os.path.join(root_path, "test_data", task_name, "results", noise_net_name)
    os.makedirs(res_path, exist_ok=True)
    if not retrain:
        save_path_str = f"vis_n{noise_net_name}" 
        save_path_str += f"_ps{pcd_size}" 
        save_path_str += f"_ndi{cfg.MODEL.NUM_DIFFUSION_ITERS}" 
        save_path_str += f"_ne{cfg.TRAIN.NUM_EPOCHS}" 
        save_path_str += f"_bs{cfg.DATALOADER.BATCH_SIZE}" 
        save_path_str += f"_ca{cfg.MODEL.INFERENCE.CANONICALIZE}"
        save_path_str += f"_ug{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].use_global_geometry}"
        save_path_str += f"_id{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].input_dim}"
        save_path_str += f"_cgf{cfg.MODEL.COND_GEOMETRY_FEATURE}"    
        save_path_str += f"_dsed{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].diffusion_step_embed_dim}"
        save_path_str += f"_up{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].use_pointnet}"
        save_path_str += f"_ro{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].rotation_orthogonalization}"
        if noise_net_name == "UNETMLP":
            save_path_str += f"_dd{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].down_dims}"
            save_path_str += f"_dpe{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].downsample_pcd_enc}"
            save_path_str += f"_ds{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].downsample_size}"
        elif noise_net_name == "TRANSFORMER":
            save_path_str += f"_nah{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].num_attention_heads}"
            save_path_str += f"_ehd{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].encoder_hidden_dim}"
            save_path_str += f"_ed{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].encoder_dropout}"
            save_path_str += f"_ea{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].encoder_activation}"
            save_path_str += f"_enl{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].encoder_num_layers}"
            save_path_str += f"_fpd{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].fusion_projection_dim}"
        elif noise_net_name == "PARALLEL_MLP":
            save_path_str += f"_fpd{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].fusion_projection_dim}"
        save_path_str += f"_uds{cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name].use_dropout_sampler}"

        save_path = os.path.join(res_path, save_path_str)
        if cfg.MODEL.INFERENCE.CANONICALIZE:
            dmorp_model.debug_inference(copy.deepcopy(train_dataset), 
                                        sample_size=cfg.MODEL.INFERENCE.SAMPLE_SIZE,
                                        consider_only_one_pair=cfg.MODEL.INFERENCE.CONSIDER_ONLY_ONE_PAIR, 
                                        debug=cfg.MODEL.INFERENCE.VISUALIZE, 
                                        shuffle=cfg.MODEL.INFERENCE.SHUFFLE,
                                        save_path=save_path,
                                        save_fig=cfg.MODEL.SAVE_FIG,
                                        visualize=cfg.MODEL.VISUALIZE,
                                        random_index=args.random_index
                                        )
        else:
            dmorp_model.debug_inference(copy.deepcopy(val_dataset), 
                            sample_size=cfg.MODEL.INFERENCE.SAMPLE_SIZE,
                            consider_only_one_pair=cfg.MODEL.INFERENCE.CONSIDER_ONLY_ONE_PAIR, 
                            debug=cfg.MODEL.INFERENCE.VISUALIZE, 
                            shuffle=cfg.MODEL.INFERENCE.SHUFFLE,
                            save_path=save_path,
                            save_fig=cfg.MODEL.SAVE_FIG,
                            visualize=cfg.MODEL.VISUALIZE,
                            random_index=args.random_index
                            )

    pass
    