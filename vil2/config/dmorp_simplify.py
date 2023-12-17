# Most simplified version of obj_dp experiment
ENV = dict(
    NUM_OBJ=8,
    NUM_STRUCTURE=4,
    SEMANTIC_FEAT_DIM=10,
    SEMANTIC_FEAT_TYPE="one_hot",  # random, clip, one_hot
)

DATALOADER = dict(
    BATCH_SIZE=1024,
    NUM_WORKERS=16,
)
TRAIN = dict(
    NUM_EPOCHS=10000,
)
MODEL = dict(
    POSE_DIM=80,  # 6d pose/ 3d translation
    GEOMETRY_FEAT_DIM=80,
    NUM_DIFFUSION_ITERS=200,
    NOISE_NET=dict(
        NAME="Transformer",
        INIT_ARGS=dict(
            input_dim=80,
            global_cond_dim=80,  # (9 (dim_feat) + 3 (pos)) * 2 (obs_horizon)
            diffusion_step_embed_dim=80,
            num_attention_heads=8,
            encoder_hidden_dim=512,
            encoder_dropout=0.1,
            encoder_num_layers=16,
            # down_dims=[1024, 2048, 2048],
            # pct_large=True,
            # nlayers=6,  # For MLP
            # hidden_size=2048,  # For MLP
        ),
    ),
    TIME_EMB_DIM=80,
    RETRAIN=False,
    PCD_SIZE=512,
    TRAIN_TEST_SPLIT=0.8,
    INFERENCE=dict(
        SAMPLE_SIZE=-1,
        CONSIDER_ONLY_ONE_PAIR=True,
        VISUALIZE=True,
        SHUFFLE=False,
        CANONICALIZE=False,
    ),
    DATASET_CONFIG = "s500-c20-r0.5", #"s500-c20-r0.5" #"s1000-c1-r0.5", # "s250-c40-r2", # "s100-c20-r2",
    SAVE_FIG=True,
    VISUALIZE=False,

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