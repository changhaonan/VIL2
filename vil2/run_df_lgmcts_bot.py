"""Run Diffusion policy on PushT scene"""
import os
import torch
import numpy as np
import collections
from tqdm.auto import tqdm
from vil2.env import env_builder
from vil2.model.dp_model import DPModel
from vil2.model.net_factory import build_vision_encoder, build_noise_pred_net
from vil2.data.dataset import DPDataset, normalize_data, unnormalize_data
import vil2.utils.misc_utils as utils
from detectron2.config import LazyConfig, instantiate


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    # load env
    task_name = "lgmcts-push_object_0"
    env = env_builder(task_name, render_mode="rgb_array")

    # build model
    cfg_file = os.path.join(root_path, "vil2", "config", "df.py")

    cfg = LazyConfig.load(cfg_file)

    vision_encoder = build_vision_encoder(cfg.MODEL.VISION_ENCODER.NAME)
    noise_pred_net = build_noise_pred_net(cfg.MODEL.NOISE_NET.NAME, **cfg.MODEL.NOISE_NET.INIT_ARGS)

    df_model = DPModel(cfg, vision_encoder=vision_encoder, noise_pred_net=noise_pred_net)

    # load dataset
    # download demonstration data from Google Drive
    dataset_path = f"{root_path}/test_data/lgmcts_bot/push_object_0.zarr"
    dataset = DPDataset(
        dataset_path=dataset_path,
        pred_horizon=cfg.MODEL.PRED_HORIZON,
        obs_horizon=cfg.MODEL.OBS_HORIZON,
        action_horizon=cfg.MODEL.ACTION_HORIZON,
    )

    stats = dataset.stats
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    df_model.train(num_epochs=cfg.TRAIN.NUM_EPOCHS, data_loader=data_loader)

    # load pre-trained
    load_pretrained = False
    if load_pretrained:
        ckpt_path = "pusht_vision_100ep.ckpt"
        state_dict = torch.load(ckpt_path, map_location=df_model.device)
        df_model.nets.load_state_dict(state_dict)
        print("Pretrained weights loaded.")
    else:
        print("Skipped pretrained weight loading.")

    ################### Start Inference ###################
    # limit enviornment interaction to 200 steps before termination
    max_steps = 200
    # use a seed > 200 to avoid initial states seen in the training dataset
    # env.seed(seed=100000)

    # get first observation
    obs, info = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque([obs] * cfg.MODEL.OBS_HORIZON, maxlen=cfg.MODEL.OBS_HORIZON)
    # save visualization and rewards
    imgs = [env.render(mode="rgb_array")]
    rewards = list()
    done = False
    step_idx = 0

    #
    with tqdm(total=max_steps, desc="Eval Push Object") as pbar:
        while not done:
            # infer action
            with torch.no_grad():
                action_pred = df_model.inference(obs_deque=obs_deque, stats=stats)
                # pred pred_length, execute action_length
                # only take action_horizon number of actions
                start = cfg.MODEL.OBS_HORIZON - 1
                end = start + cfg.MODEL.ACTION_HORIZON
                action = action_pred[start:end, :]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps without replanning
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, truncated, info = env.step(action[i])
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)
                    imgs.append(env.render(mode='rgb_array'))

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break

    # print out the maximum target coverage
    print('Score: ', max(rewards))
    video = utils.generate_video(imgs, "pusht.mp4")
