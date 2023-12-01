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
    action_horizon=2,
    obs_horizon=8,
)

POLICY = dict(
    movement_type="horizon",  # horizon, waypoint
    carrier_type="rigid_body",  # rigid_body, super_voxel, keypoint
    waypoint_num=4,
    horizon_num=4,
    act_horizon=2,
    obs_horizon=8,
)
