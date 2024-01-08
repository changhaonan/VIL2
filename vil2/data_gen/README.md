# Data generation Module

## Dmorp: Diffusion model for relative pose prediction

### Step 1: Generate random initialization

```python
python dmorp_gen.py --data_id=0 --target_object=0 --anchor_object=1 --num_samples=10000
python dmorp_gen.py --data_id=1 --target_object=1 --anchor_object=2 --num_samples=10000
```

During this process, we generated a series of random poses. It creates a series of `init_pose_1`, `init_pose_2`, `transform` pair. Here object 1 is the target object, and object 2 is the anchor object, and `transform` is required tranform of object 1 inside world coordinate.

### Step 2: Render depth & color images under generated initialization

You can parallel multiple progress. (Depending your system capcity.)

Here we have two different options, one is use `blenderproc`.
```bash
blenderproc run render_object.py --data_id=0 --num_cam_poses 20 --start_idx 0 --end_idx 250
blenderproc run render_object.py --data_id=0 --num_cam_poses 20 --start_idx 250 --end_idx 500
blenderproc run render_object.py --data_id=0 --num_cam_poses 20 --start_idx 500 --end_idx 750
blenderproc run render_object.py --data_id=0 --num_cam_poses 20 --start_idx 750 --end_idx 1000
```

The other one is to use `pyrender`.
```bash
python faster_render.py --data_id=0 --num_cam_poses 1
```
```bash
python faster_render.py --data_id=1 --num_cam_poses 1
```

### Step 3: Generate Diffusion Dataset

```bash
python preprocess_data.py --data_id=0
```

```bash
python preprocess_data.py --data_id=1
```