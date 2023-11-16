from vil2.env import env_builder


if __name__ == "__main__":
    env_name = "GYM-PointMaze_UMaze-v3"
    env = env_builder(env_name, render_mode="human", cfg={})
    env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            env.reset()