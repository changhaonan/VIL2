"""Utility functions for State Occupancy Measures (SOM)"""
import gymnasium as gym


def sample_som(env: gym.Env, policy_fn, num_steps: int, num_samples: int, seed: int = 0):
    """Sample a set of states from the environment.

    Args:
        env: Gym environment.
        policy_fn: A function that takes in a state and returns an action.
        num_samples: Number of samples to draw.
        seed: Random seed.

    Returns:
        A list of states.
    """
    env.seed(seed)
    obses = []
    for _ in range(num_samples):
        obs, info = env.reset()
        for i in range(num_steps):
            action = policy_fn(obs)
            obs, _, done, info = env.step(action)
            if done:
                break
        obses.append(obs)
    return obses
