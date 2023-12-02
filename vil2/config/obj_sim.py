ENV = dict(
    obj_names=["bowl", "plate"],
    obj_mesh_dir="google_scanned_objects",
    obj_source="google_scanned_objects",
    obj_init_poses=[
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, -0.3, 0.0, 1.0, 0.0, 0.0, 0.0],
    ],
    super_patch_size=5,
    super_patch_radius=0.05,
    max_iter=100,
    track_horizon=10,
    action_horizon=6,
    obs_horizon=8,
    img_width=96,
    img_height=96,
)

POLICY = dict(
    movement_type="horizon",  # horizon, waypoint
    carrier_type="rigid_body",  # rigid_body, super_voxel, keypoint
    waypoint_num=4,
    horizon_num=4,
    action_horizon=6,
    obs_horizon=8,
)

DATALOADER = dict(
    BATCH_SIZE=64,
    NUM_WORKERS=4,
)

MODEL = dict(
    OBS_HORIZON=2,
    ACTION_HORIZON=8,
    PRED_HORIZON=16,  # 16
    ACTION_DIM=6,  # 6d pose
    VISION_ENCODER=dict(
        NAME="resnet18",
        PRETRAINED=True,
    ),
    NOISE_NET=dict(
        NAME="UNET1D",
        INIT_ARGS=dict(
            input_dim=6,
            global_cond_dim=262,  # (128 (dim_feat) + 3 (pos)) * 2 (obs_horizon)
            diffusion_step_embed_dim=256,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
        ),
    ),
)

TRAIN = dict(
    NUM_EPOCHS=100,
)
