# VIL2

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
There seems to have encoding problem.

## Trouble-shooting

There is `cypython` problem can be found [here](https://github.com/openai/mujoco-py/issues/773).
```bash
pip install "cython<3"
```
