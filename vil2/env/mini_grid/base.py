from __future__ import annotations
import numpy as np
import heapq
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.actions import Actions
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class BaseMiniGridEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        # Define a mission (task)
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.goal_poses = []

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        """Generate everything in the scene"""
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate verical separation wall
        for i in range(0, height):
            if i == 5:
                continue
            self.grid.set(5, i, Wall())
        
        # # Place the door and key
        # self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        # self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        self.goal_poses = [(width - 2, height - 2)]

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"

    def reset(self, **kwargs):
        """Override the reset function to encode the observation"""
        obs, info = super().reset(**kwargs)
        # encode observation
        obs_encode = self.encode_obs(obs)
        return obs_encode, info

    def step(self, action):
        """Override the step function to encode the observation"""
        # process action
        action = np.rint(action.squeeze())
        obs, reward, terminated, truncated, info = super().step(action)
        # encode observation
        obs_encode = self.encode_obs(obs)
        return obs_encode, reward, terminated, truncated, info

    def encode_obs(self, observation):
        """Encode the observation into a vector"""
        image = observation["image"]
        direction = observation["direction"]
        # sinuoid encoding
        direction_encode = np.array([np.sin(direction/4.0), np.cos(direction/4.0), np.sin(2 * direction/4.0), np.cos(2 * direction/4.0)])
        encoded_obs = np.concatenate([image.flatten(), direction_encode], axis=0)
        return encoded_obs
    
    def optimal_path(self, agent_pos, goal_pos):
        """Path planning for mini-grid using A* search algorithm."""
        #TODO: currently don't consider door & key
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Initialize variables
        height, width = self.grid.height, self.grid.width
        occupancy_map = np.ones((height, width))

        # Build the occupancy map
        for i in range(height):
            for j in range(width):
                if self.grid.get(i, j) is None or isinstance(self.grid.get(i, j), Goal):
                    occupancy_map[i, j] = 0

        # Priority queue for open set
        open_set = [(0, agent_pos)]

        # Data structures for A* algorithm
        came_from = {}
        g_score = {agent_pos: 0}
        f_score = {agent_pos: heuristic(agent_pos, goal_pos)}

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal_pos:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for (dx, dy) in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Check all four neighbors
                    neighbor = (current[0] + dx, current[1] + dy)

                    if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width and occupancy_map[neighbor[0], neighbor[1]] == 0:
                        tentative_g_score = g_score[current] + 1

                        if tentative_g_score < g_score.get(neighbor, float("inf")):
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal_pos)
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # Return None if no path is found

    def get_action(self, next_pos):
        """Get the corresponding action for the next position"""
        #TODO: currently don't consider door & key
        agent_pos = self.agent_pos
        agent_dir = self.agent_dir
        next_type = self.grid.get(*next_pos)
        if next_type is None or isinstance(next_type, Goal):
            # next_type is a position or goal
            diff = np.array(next_pos) - np.array(agent_pos)
            # go up/down first
            if diff[1] < 0:
                if agent_dir == 3:
                    action = Actions.forward
                else:
                    action = Actions.left 
            elif diff[1] > 0:
                if agent_dir == 1:
                    action = Actions.forward
                else:
                    action = Actions.right
            else:
                # go left/right
                if diff[0] > 0:
                    if agent_dir == 0:
                        action = Actions.forward
                    else:
                        action = Actions.left
                elif diff[0] < 0:
                    if agent_dir == 2:
                        action = Actions.forward
                    else:
                        action = Actions.right
                else:
                    # done
                    action = None
        return action


def main():
    env = BaseMiniGridEnv(render_mode="human")
    env.reset()
    path = env.optimal_path(agent_pos=(1, 1), goal_pos=(8, 8))
    for next_pos in path:
        while True:
            action = env.get_action(next_pos)
            if action is None:
                break
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

if __name__ == "__main__":
    main()