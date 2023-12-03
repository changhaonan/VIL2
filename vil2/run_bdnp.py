"""Run Bellman Diffusion Network Policy"""
import os
import cv2
import pickle
import numpy as np
from vil2.env import env_builder
from vil2.algo.bdnp import BDNPolicy
from detectron2.config import LazyConfig


if __name__ == "__main__":
    # prepare path
    root_path = os.path.dirname((os.path.abspath(__file__)))
    env_name = "GYM-FetchReach-v2"
    export_path = os.path.join(root_path, "test_data", env_name)
    check_point_path = os.path.join(export_path, 'bdnp', 'checkpoint')
    log_path = os.path.join(export_path, 'bdnp', 'log')
    os.makedirs(export_path, exist_ok=True)
    os.makedirs(check_point_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # prepare data
    env = env_builder(env_name, render_mode="rgb_array")

    config = {
        # ----- som related -----
        'som_hidden_dim': 32,
        'som_hidden_layer': 3,
        'som_time_emb': 'learnable',
        'som_input_emb': 'learnable',
        'beta_schedule': 'squaredcos_cap_v2',
        'num_diffusion_iters': 20,
        'som_gamma': 0.3,  # balancing loss; to distinguish from the gamma for discount factor
        'som_lr': 1e-4,
        'som_weight_decay': 1e-6,
        # ----- policy related -----
        'pi_hidden_dim': 256,
        'pi_hidden_layer': 3,
        'policy_std': 0.1,
        'batch_size': 512,
        'finite_horizon': 20,
        'policy_lr': 1e-4,
        'policy_weight_decay': 1e-6,
        # ----- training related -----
        'num_epochs': 500,
        'eval_period': 1000,
        'max_epoch_per_episode': 8,
        'log_period': 100,
        'her_tolerance': 0.2,  # HER tolerance
        'alpha': 0.1,
        'target_update_period': 1,
        'enable_save': True,
        'enable_load': False,
        'log_path': log_path,
        'render_eval': True,  # render evaluation
        'lamda': 10.0,  # regularization parameter
        'sample_size': 32,
        'a_lazy_init': True,  # use last a_0 to init current a_T
    }
    # env = DictConcatWrapper(env)
    bdnp = BDNPolicy(env=env, config=config)
    # do train
    goal_pi = np.zeros((31,), dtype=np.float32)
    bdnp.train(batch_size=config['batch_size'],
               num_episode=config['num_epochs'])

    bdnp.save(os.path.join(check_point_path, 'model.pt'))

    # save the stats for replay buffer
    # stats = bdnp.replay_buffer.compute_stats()
    # with open(os.path.join(export_path, 'bdnp', 'stats.pkl'), 'wb') as f:
    #     pickle.dump(stats, f)
