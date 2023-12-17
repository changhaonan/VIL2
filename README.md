# VIL2

## DMorp

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