# Most simplified version of obj_dp experiment
ENV = dict(
    obj_names=["sphere"],
    obj_mesh_dir="google_scanned_objects",
    obj_source="virtual",
    obj_init_poses=[
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
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
    ACTION_DIM=3 + 1,  # 6d pose/ 3d translation + time stamp
    POSE_DIM=3,  # 6d pose/ 3d translation
    GEOMETRY_FEAT_DIM=128,
    NUM_DIFFUSION_ITERS=200,
    VISION_ENCODER=dict(
        NAME="resnet18",
        PRETRAINED=True,
    ),
    NOISE_NET=dict(
        NAME="UNET1D",
        INIT_ARGS=dict(
            input_dim=3+1,
            global_cond_dim=262,  # (128 (dim_feat) + 3 (pos)) * 2 (obs_horizon)
            diffusion_step_embed_dim=256,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
        ),
    ),
    RECON_VOXEL_CENTER=True,  # reconstruct voxel center
    RECON_TIME_STAMP=True,  # reconstruct time stamp
    RECON_DATA_STAMP=True,  # reconstruct data stamp
    COND_GEOMETRY_FEATURE=False,
    COND_VOXEL_CENTER=False,
    GUIDE_TIME_CONSISTENCY=True,
    GUIDE_DATA_CONSISTENCY=True,
    USE_POSITIONAL_EMBEDDING=True,
    TIME_EMB_DIM=8,
)

TRAIN = dict(
    NUM_EPOCHS=100,
)
