# Most simplified version of obj_dp experiment
ENV = dict(
    NUM_OBJ=8,
    NUM_STRUCTURE=4,
    SEMANTIC_FEAT_DIM=10,
    SEMANTIC_FEAT_TYPE="one_hot",  # random, clip, one_hot
    GOAL_TYPE="struct-m2-p1"
)

DATALOADER = dict(
    BATCH_SIZE=256,
    NUM_WORKERS=0,  # Set to 0 if using ilab
    AUGMENTATION=dict(
        IS_ELASTIC_DISTORTION=True,
        IS_RANDOM_DISTORTION=True,
        RANDOM_DISTORTION_RATE=0.2,
        RANDOM_DISTORTION_MAG=0.01,
        VOLUME_AUGMENTATION_FILE="va_rotation.yaml",  # None
    ),
)
TRAIN = dict(
    NUM_EPOCHS=10000,
    LR=1e-4,
)
MODEL = dict(
    NOISE_NET=dict(
        NAME="TRANSFORMER",
        INIT_ARGS=dict(
            TRANSFORMER=dict(
                pcd_input_dim=9,  # 3 + 3 + 3
                pcd_output_dim=256,  # (16, 32, 64, 128)
                use_pcd_mean_center=True,
                num_attention_heads=8,
                encoder_hidden_dim=512,
                encoder_dropout=0.0,
                encoder_num_layers=8,
                obj_dropout=0.1,
            ),
        ),
    ),
    TIME_EMB_DIM=128,
    RETRAIN=True,
    PCD_SIZE=512,
    TRAIN_SPLIT=0.7,
    VAL_SPLIT=0.2,
    TEST_SPLIT=0.1,
    DATASET_CONFIG="s25000-c1-r0.5",  # "s1000-c200-r0.5",  # "s300-c20-r0.5", #"s500-c20-r0.5" #"s1000-c1-r0.5", # "s250-c40-r2", # "s100-c20-r2",
)
LOGGER = dict(
    PROJECT="tns",
)
CUDA_DEVICE = "cuda"
