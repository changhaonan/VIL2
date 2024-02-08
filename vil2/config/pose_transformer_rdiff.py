# Most simplified version of obj_dp experiment
ENV = dict(
    NUM_OBJ=8,
    NUM_STRUCTURE=4,
    SEMANTIC_FEAT_DIM=10,
    SEMANTIC_FEAT_TYPE="one_hot",  # random, clip, one_hot
    GOAL_TYPE="rdiff",  # rdiff
)
PREPROCESS = dict(
    GRID_SIZE=0.025,
    TARGET_RESCALE=3.0,
    NUM_POINT_LOW_BOUND=30,
    NUM_POINT_HIGH_BOUND=400,
)

DATALOADER = dict(
    BATCH_SIZE=32,
    NUM_WORKERS=8,  # Set to 0 if using ilab
    AUGMENTATION=dict(
        IS_ELASTIC_DISTORTION=True,
        IS_RANDOM_DISTORTION=True,
        RANDOM_DISTORTION_RATE=0.2,
        RANDOM_DISTORTION_MAG=0.01,
        VOLUME_AUGMENTATION_FILE="va_rotation.yaml",  # None
        RANDOM_SEGMENT_DROP_RATE=0.15,
        CROP_PCD=True,
        CROP_SIZE=0.2,
        CROP_NOISE=0.1,
        CROP_STRATEGY="knn",  # bbox, radius, knn
        NOISE_LEVEL=0.5,
        ROT_AXIS="yz",
        KNN_K=20,
    ),
)
TRAIN = dict(
    NUM_EPOCHS=10000,
    WARM_UP_STEP=0,
    LR=1e-4,
    GRADIENT_CLIP_VAL=1.0,
)
MODEL = dict(
    DIFFUSION_PROCESS="ddpm",
    NUM_DIFFUSION_ITERS=100,
    NOISE_NET=dict(
        NAME="TRANSFORMERV2",
        INIT_ARGS=dict(
            TRANSFORMER=dict(
                pcd_input_dim=6,  # 3 + 3 + 3
                pcd_output_dim=512,  # (16, 32, 64, 128)
                use_pcd_mean_center=True,
                points_pyramid=[128, 32],
                num_attention_heads=8,
                encoder_hidden_dim=256,
                encoder_dropout=0.1,
                encoder_activation="relu",
                encoder_num_layers=2,
                fusion_projection_dim=256,
                use_semantic_label=True,
                translation_only=True,
            ),
            TRANSFORMERV2=dict(
                grid_sizes=[0.035, 0.06],
                depths=[2, 3, 3],
                dec_depths=[1, 1],
                hidden_dims=[64, 128, 256],
                n_heads=[4, 8, 8],
                ks=[16, 24, 32],
                in_dim=6,
                fusion_projection_dim=256,
            ),
        ),
    ),
    TIME_EMB_DIM=128,
    RETRAIN=True,
    # PCD_SIZE=512,
    PCD_SIZE=2048,
    TRAIN_SPLIT=0.7,
    VAL_SPLIT=0.2,
    TEST_SPLIT=0.1,
    INFERENCE=dict(
        SAMPLE_SIZE=-1,
        CONSIDER_ONLY_ONE_PAIR=False,
        VISUALIZE=False,
        SHUFFLE=False,
        CANONICALIZE=False,
    ),
    DATASET_CONFIG="s25000-c1-r0.5",  # "s1000-c200-r0.5",  # "s300-c20-r0.5", #"s500-c20-r0.5" #"s1000-c1-r0.5", # "s250-c40-r2", # "s100-c20-r2",
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
    GMM=dict(
        N_COMPONENTS=5,
    ),
)
LOGGER = dict(
    PROJECT="tns",
)
CUDA_DEVICE = "cuda"
