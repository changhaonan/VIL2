# Most simplified version of obj_dp experiment
ENV = dict(
    NUM_OBJ=8,
    NUM_STRUCTURE=4,
    SEMANTIC_FEAT_DIM=10,
    SEMANTIC_FEAT_TYPE="one_hot",  # random, clip, one_hot
)

DATALOADER = dict(
    BATCH_SIZE=128,
    # NUM_WORKERS=8,
)
TRAIN = dict(
    NUM_EPOCHS=10000,
)
MODEL = dict(
    POSE_DIM=256,  # 6d pose/ 3d translation
    GEOMETRY_FEAT_DIM=1024,
    NUM_DIFFUSION_ITERS=200,
    NOISE_NET=dict(
        NAME="TRANSFORMER",
        INIT_ARGS=dict(
            UNETMLP = dict(
                input_dim=256,
                global_cond_dim=1024, 
                diffusion_step_embed_dim=128,
                use_global_geometry=True,
                use_pointnet=False,
                use_dropout_sampler=False,
                rotation_orthogonalization=True,
                down_dims=[256, 512, 1024], #[1024, 2048, 2048]
                downsample_pcd_enc=False,
                downsample_size=256,
            ),
            TRANSFORMER = dict(
                input_dim=256,
                global_cond_dim=1024, 
                diffusion_step_embed_dim=128,
                use_global_geometry=True,
                use_pointnet=False,
                rotation_orthogonalization=False,
                num_attention_heads=2,
                encoder_hidden_dim=512,
                encoder_dropout=0.1,
                encoder_activation='relu',
                encoder_num_layers=2,
                fusion_projection_dim=256,
                downsample_pcd_enc=False,
                downsample_size=128,
                use_dropout_sampler=False,
            ),
            PARALLELMLP = dict(
                input_dim=256,
                global_cond_dim=1024, 
                diffusion_step_embed_dim=128,
                use_global_geometry=True,
                use_pointnet=False,
                use_dropout_sampler=False,
                rotation_orthogonalization=False,
                fusion_projection_dim=512,
                downsample_pcd_enc=False,
                downsample_size=256,
            )
        ),
    ),
    TIME_EMB_DIM=128,
    RETRAIN=True,
    PCD_SIZE=512,
    TRAIN_TEST_SPLIT=0.8,
    INFERENCE=dict(
        SAMPLE_SIZE=-1,
        CONSIDER_ONLY_ONE_PAIR=False,
        VISUALIZE=False,
        SHUFFLE=False,
        CANONICALIZE=False,
    ),
    DATASET_CONFIG = "s300-c20-r0.5", # "s300-c20-r0.5", #"s500-c20-r0.5" #"s1000-c1-r0.5", # "s250-c40-r2", # "s100-c20-r2",
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