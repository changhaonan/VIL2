# VIL2

## DMorp

### Installation
> Step 0:
```
conda create -n mcts python=3.8
```
> Step 1:
```
pip install -r base_requirements.txt
pip install -e .
pip install git+https://github.com/marian42/mesh_to_sdf.git
pip install git+https://github.com/facebookresearch/detectron2.git
```

> Step2: If using CUDA 11.8, install pytorch and other related modules via the following:
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html --no-index
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html --no-index
pip install https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install git+https://github.com/facebookresearch/pytorch3d.git
```
> Step 3: Setting the env variables
```
export RPDIFF_SOURCE_DIR=$PWD/vil2/external/rpdiff
```

### Progress

- [x] Fit two objects data.
- [ ] Fit multi-object data, up to 5.
- [ ] Fit to multi-modality data.
- [ ] Combine with MCTS sampler.
- [ ] Connect with real-data pipeline.


## Notice

```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```


## Env

### Maze

`2D maze` implemented to capture the multi-modality kernel of the problem.

### Isaac

## TODO:

- [ ] Finish the orcale motion and finish decision
- [ ] Add imitation learning data generation
- [ ] IL
- [ ] Offline-RL
- [ ] Diffusion-policies
- [ ] Max-diff RL
- [ ] Further check the detail of this module.

#### BDNP
- [ ] Normalize data & action.

## OBJDP
There seems to have encoding problem. The most important now is to do encoding.

## Trouble-shooting

1. Mujoco install
There is `cypython` problem can be found [here](https://github.com/openai/mujoco-py/issues/773).
```bash
pip install "cython<3"
```

2. BlenderProc
imgcodecs: OpenEXR codec is disabled. You can enable it via 'OPENCV_IO_ENABLE_OPENEXR' option.
```bash
export OPENCV_IO_ENABLE_OPENEXR=true
```