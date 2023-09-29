import cv2
from vil2.env import ENV_MAP
from stable_baselines3 import PPO


if __name__ == "__main__":
    env_name = "maze"
    config = {
        "seed": 0,
        "num_level": 5, 
        "num_branch": 2,
        "num_goal": 3,
        "end_probs": [0.0, 0.0, 0.0, 0.3, 0.3, 0.0],
        "noise_level": 0.0,
    }
    env = ENV_MAP[env_name](config=config)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=35000)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = vec_env.step(action)
        print(f"step {i}: {reward}, {terminated}, {info}")
        images = vec_env.get_images()
        cv2.imshow("image", images[0])
        cv2.waitKey(1)
        if terminated:
          obs = vec_env.reset()

    env.close()