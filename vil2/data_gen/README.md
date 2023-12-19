# Data generation Module

## Dmorp: Diffusion model for relative pose prediction

### Step 1: Generate random initialization

```python
python dmorp_gen.py --data_id=0 --target_object=tea_pot --anchor_object=tea_mug
python dmorp_gen.py --data_id=1 --target_object=spoon --anchor_object=tea_pot
```

During this process, we generated a series of random poses. It creates a series of `init_pose_1`, `init_pose_2`, `transform` pair. Here object 1 is the target object, and object 2 is the anchor object, and `transform` is required tranform of object 1 inside world coordinate.

### Step 2: Render depth & color images under generated initialization

```bash
blenderproc run render_object.py --num_cam_poses 4
```

### Step 3: Generate Diffusion Dataset

```bash
python data_loader.py
```