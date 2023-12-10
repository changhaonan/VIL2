# Most simplified version of obj_dp experiment
ENV = dict(
    NUM_OBJ=8,
    NUM_STRUCTURE=4,
    SEMANTIC_FEAT_DIM=16,
    SEMANTIC_FEAT_TYPE="one_hot",  # random, clip, one_hot
)

DATALOADER = dict(
    BATCH_SIZE=32,
    NUM_WORKERS=4,
)

MODEL = dict(
    MAX_SCENE_SIZE=4,
    ACTION_DIM=3 + 1,  # 6d pose/ 3d translation + time stamp
    POSE_DIM=7,  # 6d pose/ 3d translation
    GEOMETRY_FEAT_DIM=128,
    SEMANTIC_FEAT_DIM=16,
    NUM_DIFFUSION_ITERS=200,
    VISION_ENCODER=dict(
        NAME="resnet18",
        PRETRAINED=True,
    ),
    NOISE_NET=dict(
        NAME="UNETMLP",
        INIT_ARGS=dict(
            input_dim=3+1,
            global_cond_dim=262,  # (128 (dim_feat) + 3 (pos)) * 2 (obs_horizon)
            diffusion_step_embed_dim=256,
            down_dims=[1024, 2048, 2048],
            # nlayers=6,  # For MLP
            # hidden_size=2048,  # For MLP
        ),
    ),
    RECON_DATA_STAMP=True,  # reconstruct data stamp
    RECON_SEMANTIC_FEATURE=True,  # reconstruct semantic feature
    RECON_POSE=True,  # reconstruct pose
    COND_GEOMETRY_FEATURE=False,
    COND_SEMANTIC_FEATURE=False,
    GUIDE_DATA_CONSISTENCY=True,
    GUIDE_SEMANTIC_CONSISTENCY=False,
    USE_POSITIONAL_EMBEDDING=True,
    TIME_EMB_DIM=16,
    AGGREGATE_TYPE="sum",  # mean, max, sum
    AGGREGATE_LIST=["SEMANTIC", "POSE"],
    SEMANTIC_FEAT_TYPE="one_hot",  # random, clip, one_hot
)

TRAIN = dict(
    NUM_EPOCHS=500,
)
