"""A example env of 2d maze"""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import gym

class MazeTree(gym.Env):
    """A maze tree is a tree with multiple levels and multiple branches."""
    def __init__(self, num_level: int, num_branch: int, num_goal: int, seed: int = 0):
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.num_level = num_level
        self.num_branch = num_branch
        self.num_goal = num_goal
        self.end_probs = [0.0, 0.0, 0.0, 0.4, 0.5]  # probability of ending
        self.dynamic_prob = 0.0  # probability of dynamic
        if len(self.end_probs) < self.num_level:
            self.end_probs.extend([0.2] * (self.num_level - len(self.end_probs)))
        self.reset()

    def step(self, action):
        # check the cur direction
        cur_dir = self.node_dir[self.cur_node]
        action = (action - cur_dir) % 1.0
        # go to the next node
        random_val = self.rng.random()
        if random_val < self.dynamic_prob:
            # go to a random node
            self.cur_node = self.rng.choice(self.node_children[self.cur_node] + self.node_parent[self.cur_node])
        else:
            # static
            num_dir = len(self.node_children[self.cur_node] + self.node_parent[self.cur_node])
            dir_idx = int(action * num_dir) % num_dir
            if dir_idx < len(self.node_children[self.cur_node]):
                self.cur_node = self.node_children[self.cur_node][dir_idx]
            else:
                self.cur_node = self.node_parent[self.cur_node][dir_idx - len(self.node_children[self.cur_node])]
        # check if the cur_node is goal
        done = self.cur_node in self.goal_list
        # get the reward
        reward = 1.0 if done else 0.0
        return self.cur_node, reward, done, {}

    def reset(self):
        """Reset the maze tree"""
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
            print(f"Expanding layer {num_layer}...")
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
        self.goal_list = self.rng.choice(cur_node_list, self.num_goal, replace=False)
        # set cur node to 0
        self.cur_node = 0
        return True

    def render(self):
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
        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, node_color=color_list, with_labels=False)
        plt.axis("equal")
        plt.show()


if __name__  == "__main__":
    maze = MazeTree(num_level=5, num_branch=2, num_goal=4, seed=1)
    maze.render()

    for i in range(100):
        action = maze.rng.random()
        next_node, reward, done, _ = maze.step(action)
        maze.render()
        print(f"Step {i}: action={action}, next_node={next_node}, reward={reward}")
        if done:
            break