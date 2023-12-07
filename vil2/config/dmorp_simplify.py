# Most simplified version of obj_dp experiment
ENV = dict(
    NUM_OBJ=8,
    NUM_STRUCTURE=4,
    SEMANTIC_FEAT_DIM=512,
    SEMANTIC_FEAT_TYPE="clip",  # random, clip, one_hot
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
    MAX_SCENE_SIZE=4,
    OBS_HORIZON=2,
    ACTION_HORIZON=8,
    PRED_HORIZON=16,  # 16
    ACTION_DIM=3 + 1,  # 6d pose/ 3d translation + time stamp
    POSE_DIM=3,  # 6d pose/ 3d translation
    GEOMETRY_FEAT_DIM=128,
    SEMANTIC_FEAT_DIM=512,
    NUM_DIFFUSION_ITERS=200,
    VISION_ENCODER=dict(
        NAME="resnet18",
        PRETRAINED=True,
    ),
    NOISE_NET=dict(
        NAME="MLP",
        INIT_ARGS=dict(
            input_dim=3+1,
            global_cond_dim=262,  # (128 (dim_feat) + 3 (pos)) * 2 (obs_horizon)
            diffusion_step_embed_dim=256,
            down_dims=[1024, 2048, 2048],
        ),
    ),
    RECON_DATA_STAMP=False,  # reconstruct data stamp
    RECON_SEMANTIC_FEATURE=True,  # reconstruct semantic feature
    RECON_POSE=False,  # reconstruct pose
    COND_GEOMETRY_FEATURE=False,
    COND_SEMANTIC_FEATURE=False,
    GUIDE_DATA_CONSISTENCY=False,
    GUIDE_SEMANTIC_CONSISTENCY=False,
    USE_POSITIONAL_EMBEDDING=True,
    TIME_EMB_DIM=128,
    SEMANTIC_FEAT_TYPE="clip",  # random, clip, one_hot
)

TRAIN = dict(
    NUM_EPOCHS=1000,
)
