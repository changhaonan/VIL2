# Most simplified version of obj_dp experiment
ENV = dict(
    NUM_OBJ=8,
    NUM_STRUCTURE=4,
    SEMANTIC_FEAT_DIM=10,
    SEMANTIC_FEAT_TYPE="one_hot",  # random, clip, one_hot
)

DATALOADER = dict(
    BATCH_SIZE=32,
    NUM_WORKERS=4,
)
TRAIN = dict(
    NUM_EPOCHS=500,
)
MODEL = dict(
    POSE_DIM=80,  # 6d pose/ 3d translation
    GEOMETRY_FEAT_DIM=80,
    NUM_DIFFUSION_ITERS=100,
    NOISE_NET=dict(
        NAME="UNETMLP",
        INIT_ARGS=dict(
            input_dim=80,
            global_cond_dim=80,  # (9 (dim_feat) + 3 (pos)) * 2 (obs_horizon)
            diffusion_step_embed_dim=80,
            # num_attention_heads=8,
            # encoder_hidden_dim=256,
            # encoder_dropout=0.1,
            # encoder_num_layers=8,
            # down_dims=[1024, 2048, 2048],
            # pct_large=True,
            # nlayers=6,  # For MLP
            # hidden_size=2048,  # For MLP
        ),
    ),
    TIME_EMB_DIM=80,
    RETRAIN=True,
    PCD_SIZE=512,
    TRAIN_TEST_SPLIT=1.0,
    INFERENCE=dict(
        SAMPLE_SIZE=560,
        CONSIDER_ONLY_ONE_PAIR=False,
        VISUALIZE=True,
        SHUFFLE=True,
        CANONICALIZE=True,
    ),
    DATASET_CONFIG = "s250-c40-r2", # "s100-c20-r2",
    SAVE_FIG=True,
    VISUALIZE=True,

    MAX_SCENE_SIZE=4,
    ACTION_DIM=3 + 1,  # 6d pose/ 3d translation + time stamp
    VISION_ENCODER=dict(
    NAME="resnet18",
    PRETRAINED=True,
    ),
    SEMANTIC_FEAT_DIM=10,
    AGGREGATE_TYPE="sum",  # mean, max, sum
    AGGREGATE_LIST=["POSE"],
    SEMANTIC_FEAT_TYPE="one_hot",  # random, clip, one_hot,
    RECON_DATA_STAMP=False,  # reconstruct data stamp
    RECON_SEMANTIC_FEATURE=False,  # reconstruct semantic feature
    RECON_POSE=True,  # reconstruct pose
    COND_GEOMETRY_FEATURE=True,
    COND_SEMANTIC_FEATURE=False,
    GUIDE_DATA_CONSISTENCY=True,
    GUIDE_SEMANTIC_CONSISTENCY=False,
    USE_POSITIONAL_EMBEDDING=True,

)



CUDA_DEVICE = "cuda"