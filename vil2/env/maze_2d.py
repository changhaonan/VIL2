"""A example env of 2D maze"""
from __future__ import annotations
import cv2
import gymnasium as gym
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import vil2.utils.misc_utils as misc_utils

DIM_OBS = 512
MAX_NUM_NODES = 1000

class MazeTree(gym.Env):
    render_mode = "rgb_array"

    """A maze tree is a tree with multiple levels and multiple branches."""
    def __init__(self, config: dict[str, any]):
        """Initialize the maze tree"""
        seed = config.get("seed", 0)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.num_level = config.get("num_level", 5)
        self.num_branch = config.get("num_branch", 2)
        self.num_goal = config.get("num_goal", 3)
        self.end_probs = config.get("end_probs", [0.0, 0.0, 0.0, 0.4, 0.5])
        self.max_step = config.get("max_step", 100)
        self.noise_level = config.get("noise_level", 0.0)  # probability of dynamic
        if len(self.end_probs) < self.num_level:
            self.end_probs.extend([0.2] * (self.num_level - len(self.end_probs)))
        # encoding
        self.position_encoding = misc_utils.get_positional_encoding(MAX_NUM_NODES, DIM_OBS)
        # reset
        self.reset(reset_maze=True)

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(DIM_OBS, ), dtype=np.float32)

    @property    
    def action_space(self):
        return gym.spaces.Box(low=0.0, high=1.0, shape=(1, ), dtype=np.float32)

    def get_obs(self):
        # encode the current node
        return self.position_encoding[self.cur_node]

    def decode_obs(self, obs):
        # decode the obs to node
        return np.argmax(np.matmul(self.position_encoding, obs))
    
    def get_info(self):
        return {}

    def step(self, action):
        # check the cur direction
        cur_dir = self.node_dir[self.cur_node]
        action = (action - cur_dir) % 1.0
        # go to the next node
        noise = (self.rng.random() * 2.0 - 1.0) * self.noise_level
        action += noise
        # static
        num_dir = len(self.node_children[self.cur_node] + self.node_parent[self.cur_node])
        dir_idx = int(action * num_dir) % num_dir
        if dir_idx < len(self.node_children[self.cur_node]):
            self.cur_node = self.node_children[self.cur_node][dir_idx]
        else:
            # go back
            self.cur_node = self.node_parent[self.cur_node][dir_idx - len(self.node_children[self.cur_node])]
        # check if the cur_node is goal
        done = self.cur_node in self.goal_list
        self.step_count += 1
        if self.step_count >= self.max_step:
            truncated = True
        else:
            truncated = False
        # get the reward
        if truncated:
            reward = -1.0
        elif done:
            reward = 10.0
        else:
            reward = -0.1
        done = done or truncated
        return self.get_obs(), reward, done, truncated, self.get_info()

    def reset(self, seed=None, **kwargs):
        """Reset the maze tree"""
        reset_maze = kwargs.get("reset_maze", False)
        if reset_maze:
            # maze-tree is a tree with level height and branch width
            self.node_list = []
            self.node_children = {}
            self.node_parent = {}
            self.node_layer = {}
            self.node_dir = {}  # the direction of the node

            # layer zero
            num_layer = 1
            self.num_nodes = 0
            self.num_edges = 0
            self.node_list.append(0)
            self.num_nodes += 1
            self.node_parent[0] = []
            self.node_children[0] = []
            
            # each node will expand num_branch nodes; with probability end_prob, it will end; otherwise, it will expand
            # using BFS to expand the tree
            cur_node_list = [0]
            self.node_layer[0] = 0
            self.node_dir[0] = self.rng.random()
            while num_layer < self.num_level:
                # expand each node in the current layer
                layer_node_list = []
                while len(cur_node_list) > 0:
                    node = cur_node_list.pop(0)
                    rand_val = self.rng.random()
                    # print(f"{rand_val} vs {self.end_probs[num_layer]}")
                    if rand_val < self.end_probs[num_layer]:
                        continue
                    # expand the node
                    for i in range(self.num_branch):
                        self.node_list.append(self.num_nodes)
                        self.node_layer[self.num_nodes] = num_layer
                        self.node_dir[self.num_nodes] = self.rng.random()
                        layer_node_list.append(self.num_nodes)
                        self.node_children[node].append(self.num_nodes)
                        self.node_children[self.num_nodes] = []
                        self.node_parent[self.num_nodes] = [node]
                        self.num_nodes += 1
                        self.num_edges += 1
                cur_node_list = layer_node_list
                num_layer += 1

            # randomly select num_goal nodes as goal
            self.goal_list = self.rng.choice(cur_node_list, min(self.num_goal, len(cur_node_list)), replace=False)
        # set cur node to 0
        self.cur_node = 0
        self.step_count = 0
        return self.get_obs(), self.get_info()

    def render(self, mode="rgb_array"):
        # visualize the maze-tree
        G = nx.DiGraph()
        for node in self.node_list:
            G.add_node(node, layer=self.node_layer[node])
        for node in self.node_children:
            for child in self.node_children[node]:
                G.add_edge(node, child)
        pos = nx.multipartite_layout(G, subset_key="layer")
        # color for node is blue, the dead-end is red, the goal is green
        color_list = []
        for node in self.node_list:
            if node == self.cur_node:
                color_list.append("yellow")
            elif node in self.goal_list:
                color_list.append("green")
            elif len(self.node_children[node]) == 0:
                color_list.append("red")
            else:
                color_list.append("blue")
        
        # draw the graph
        fig = plt.figure(figsize=(8, 8))
        nx.draw(G, pos, node_color=color_list, with_labels=False)
        plt.axis("equal")
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure to free up memory
        buf.seek(0)
        img_array = np.asarray(bytearray(buf.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        return img

    def get_image(self):
        return self.render()


if __name__  == "__main__":
    config = {
        "seed": 0,
        "num_level": 5, 
        "num_branch": 2,
        "num_goal": 3,
        "end_probs": [0.0, 0.0, 0.0, 0.3, 0.3, 0.0],
        "noise_level": 0.0,
    }
    maze = MazeTree(config)
    maze.render()

    for i in range(100):
        action = maze.rng.random()
        obs, reward, terminated, _, info = maze.step(action)
        image = maze.render()
        # print(f"Step {i}: action={action}, next_node={obs}, reward={reward}")
        cv2.imshow("maze", image)
        cv2.waitKey(0)
        # if done:
        #     break