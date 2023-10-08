import d4rl
import gym
import os

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
check_point_path = os.path.join(root_path, 'checkpoint')
os.makedirs(check_point_path, exist_ok=True)
# Create the environment
env = gym.make('maze2d-umaze-v1')

# d4rl abides by the OpenAI gym interface
env.reset()
for i in range(1000):
    env.step(env.action_space.sample())
    env.render()
