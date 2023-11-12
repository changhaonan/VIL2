from __future__ import annotations
import numpy as np
import heapq
from copy import deepcopy
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.actions import Actions
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
import warnings
import random


class BaseMiniGridEnv(MiniGridEnv):
    """Basic wrapper for MiniGridEnv"""

    def __init__(
        self,
        grid_size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        goal_poses=[],
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = tuple(agent_start_pos)
        self.agent_start_dir = agent_start_dir

        # Define a mission (task)
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.goal_poses = [tuple(goal_pos) for goal_pos in goal_poses]
        if max_steps is None:
            max_steps = 4 * grid_size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
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
            self.grid.set(3, i, Wall())
            self.grid.set(5, i, Wall())
            self.grid.set(7, i, Wall())

        # Place the door and key
        self.grid.set(5, 5, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 5, Door(COLOR_NAMES[1], is_locked=True))
        self.grid.set(7, 5, Door(COLOR_NAMES[1], is_locked=True))
        self.grid.set(2, 6, Key(COLOR_NAMES[1]))
        self.grid.set(4, 6, Key(COLOR_NAMES[0]))

        # Place a goal square
        for goal_pos in self.goal_poses:
            self.put_obj(Goal(), goal_pos[0], goal_pos[1])

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"

    def reset(self, **kwargs):
        """Override the reset function to encode the observation"""
        goal_poses = kwargs.pop("goal_poses", [])
        if len(goal_poses) > 0:
            self.goal_poses = goal_poses
        agent_start_pos = kwargs.pop("agent_start_pos", None)
        if agent_start_pos is not None:
            self.agent_start_pos = agent_start_pos
        agent_start_dir = kwargs.pop("agent_start_dir", None)
        if agent_start_dir is not None:
            self.agent_start_dir = agent_start_dir
        obs, info = super().reset(**kwargs)
        # encode observation
        obs_encode = self.encode_obs(obs)
        return obs_encode, info

    def step(self, action):
        """Override the step function to encode the observation"""
        # process action
        if isinstance(action, np.ndarray):
            action = np.rint(action.squeeze()).astype(int)

        obs, reward, terminated, truncated, info = super().step(action)
        # encode observation
        obs_encode = self.encode_obs(obs)
        return obs_encode, reward, terminated, truncated, info

    def encode_obs(self, observation):
        """Encode the observation into a vector"""
        image = observation["image"]
        direction = observation["direction"]
        # sinuoid encoding
        direction_encode = np.array([np.sin(direction/4.0), np.cos(
            direction/4.0), np.sin(2 * direction/4.0), np.cos(2 * direction/4.0)])
        encoded_obs = np.concatenate(
            [image.flatten(), direction_encode], axis=0)
        return encoded_obs

    def generate_occupancy_map(self):
        # Initialize variables
        height, width = self.grid.height, self.grid.width
        occupancy_map = np.ones((height, width)).astype(int)

        # color-keylocation map
        key_info = {}

        # Build the occupancy map
        for i in range(height):
            for j in range(width):
                grid_object = self.grid.get(i, j)
                if grid_object is None or isinstance(grid_object, Goal):
                    occupancy_map[i, j] = 0
                elif isinstance(grid_object, Door):
                    # Not wall, is Door, add lock info, add color info
                    occupancy_map[i, j] = (1 << 1) | (grid_object.is_locked << 2) | (
                        COLOR_NAMES.index(grid_object.color) << 3)
                elif isinstance(grid_object, Key):
                    # Not wall, is Key, add color info
                    occupancy_map[i, j] = (1 << 2) | (
                        COLOR_NAMES.index(grid_object.color) << 3)
                    key_info[grid_object.color] = (i, j)

        return occupancy_map, key_info

    def compute_doors_to_goal(self, agent_pos, goal_pos, occupancy_map):
        doors_to_goal = []
        doors_set = set()

        def can_go_collect_doors(neighbor, value, args):
            if value == 1 or value & 0b110 == 0b100:
                return False
            if value & 0b110 == 0b110 and neighbor not in args["doors_set"]:
                color = COLOR_NAMES[(value & 0b111000) >> 3]
                args["doors_array"].append((color, neighbor))
                args["doors_set"].add(neighbor)
            return True
        args = {"can_go": {"doors_array": doors_to_goal, "doors_set": doors_set}}
        path = self.modular_a_star(
            agent_pos, goal_pos, can_go=can_go_collect_doors, occupancy_map=occupancy_map, args=args)
        return doors_to_goal, path

    def modular_a_star(self, agent_pos, goal_pos, can_go=None, heuristic=None, score=None, args=None,  occupancy_map=None):
        """Path planning for mini-grid using A* search algorithm."""
        def default_heuristic(a, b, args):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def default_score(current_score, neighbor, args):
            return current_score + 1

        # Initialize variables
        height, width = self.grid.height, self.grid.width

        if occupancy_map is None:
            occupancy_map = np.ones((height, width))
            # Build the occupancy map
            for i in range(height):
                for j in range(width):
                    if self.grid.get(i, j) is None or isinstance(self.grid.get(i, j), Goal):
                        occupancy_map[i, j] = 0

        # Priority queue for open set
        open_set = [(0, agent_pos)]

        heuristic = default_heuristic if heuristic is None else heuristic
        score = default_score if score is None else score

        # Data structures for A* algorithm
        came_from = {}
        g_score = {agent_pos: 0}
        f_score = {agent_pos: heuristic(
            agent_pos, goal_pos, args.get("heuristic", None))}

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
                if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width:
                    if can_go is None and occupancy_map[neighbor[0], neighbor[1]] == 1:
                        continue
                    if can_go is not None and not can_go(neighbor, occupancy_map[neighbor[0], neighbor[1]], args["can_go"]):
                        continue

                    tentative_g_score = score(
                        g_score[current], neighbor, args.get("score", None))

                    if tentative_g_score < g_score.get(neighbor, float("inf")):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + \
                            heuristic(neighbor, goal_pos,
                                      args.get("heuristic", None))
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # Return None if no path is found

    def optimal_action_trajectory(self, goal_pos, random_action_prob=0.0, max_steps=None, seed=None):
        """Find the optimal action trajectory to reach the goal from start position"""
        def not_reached_target(goal_type, goal_pos):
            # checks if the agent is
            #       - at goal_pos, if goal_type is goal
            #       - around goal_pos (1 block away), elsewhere
            return (goal_type == "goal" and self.agent_pos != goal_pos) or (goal_type != "goal" and np.sum(np.abs(np.array(goal_pos) - np.array(self.agent_pos))) > 1)

        def no_go_doors(neighbor, value, args):
            # custom can_go function that blocks doors & keys
            if neighbor == args["target_pos"]:
                return True
            if value == 1 or value & 0b110 == 0b100 or value & 0b110 == 0b110:
                return False
            return True

        def execute_action(action):
            # self.step wrapper to build action trajectory
            self.step(action)
            if max_steps is not None and len(action_trajectory) == max_steps:
                return -1
            action_trajectory.append(action)

        def generate_mini_goals_from_path(path, goal_pos, occupancy_map, key_info):
            minigoals = []
            for entry in path:
                value = occupancy_map[entry]
                if value & 0b110 == 0b110:
                    # door
                    color = COLOR_NAMES[(value & 0b111000) >> 3]
                    if color not in key_info:
                        warnings.warn(
                            "Not enough keys to solve this grid! Returning empty trajectory.")
                        return []
                    minigoals.append({
                        "type": "key",
                        "color": color
                    })
                    minigoals.append({
                        "type": "door",
                        "color": color,
                        "location": entry
                    })
            minigoals.append({
                "type": "goal",
                "location": goal_pos
            })
            return minigoals

        def weighted_score(current_score, neighbor, args):
            value = args["occupancy_map"][neighbor]
            if value & 0b110 == 0b100:
                # key
                return current_score + 2
            if value & 0b110 == 0b110:
                # door
                return current_score + 5
            return current_score + 1

        prev_render_mode = self.render_mode
        # self.render_mode = "none"
        action_trajectory = []  # FIXME: such writing method is not closed
        occupancy_map, key_info = self.generate_occupancy_map()

        path = self.modular_a_star(self.agent_pos, goal_pos, occupancy_map=occupancy_map, score=weighted_score, args={
                                   "score": {"occupancy_map": occupancy_map}})
        minigoals = generate_mini_goals_from_path(
            path, goal_pos, occupancy_map, key_info)

        for goal_index, goal in enumerate(minigoals):
            target_pos = goal["location"] if goal["type"] != "key" else key_info[goal["color"]]
            done = False
            # done is set to true only when
            #     - key is picked
            #     - door is opened
            #     - Goal is reached
            while not done:
                path = self.modular_a_star(agent_pos=self.agent_pos, goal_pos=target_pos, occupancy_map=occupancy_map, can_go=no_go_doors, args={
                                           "can_go": {"target_pos": target_pos}})
                if path is None:
                    warnings.warn(
                        "Cannot Solve this maze! Returning empty trajectory!")
                    return []
                # navigate upto the goal with random actions
                for next_pos in path:
                    random_action = False
                    while True:
                        action, random_action = self.get_action(
                            next_pos, random_action_prob)
                        if random_action:
                            stat = execute_action(action)
                            if stat == -1:
                                return action_trajectory

                            break
                        if next_pos == path[-1] and action == Actions.forward and goal["type"] != "goal":
                            break
                        if action is None:
                            break
                        stat = execute_action(action)
                        if stat == -1:
                            return action_trajectory
                    if random_action:
                        break
                if goal["type"] == "goal" and not not_reached_target(goal["type"], target_pos):
                    # reached goal target
                    done = True
                    break
                 # we are at the target and now we try to turn towards it, this loop only goes on as long as we are one box away from the goal_pos
                while not not_reached_target(goal["type"], target_pos):
                    action, random_action = self.get_action(
                        target_pos, random_action_prob)
                    if action == Actions.forward:
                        # we are still one block away and we are facing the door/ key
                        if goal["type"] == "key":
                            action = Actions.pickup
                        elif goal["type"] == "door":
                            action = Actions.toggle
                        elif goal["type"] == "drop":
                            action = Actions.drop
                        stat = execute_action(action)
                        if stat == -1:
                            return action_trajectory
                        done = True
                        # drop the key after unlocking a door
                        if goal["type"] == "door" and minigoals[goal_index+1]["type"] == "key" and minigoals[goal_index+1]["color"] != goal["color"]:
                            # drop key around the agent location
                            drop_loc = None
                            for (dx, dy) in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                pos = (self.agent_pos[0] +
                                       dx, self.agent_pos[1] + dy)
                                if pos == path[-1]:
                                    continue
                                item = self.grid.get(*pos)
                                if item is None or item.can_contain():
                                    drop_loc = pos
                                    # this is our new mini-goal to drop the key
                                    minigoals.insert(goal_index+1, {
                                        "type": "drop",
                                        "location": drop_loc
                                    })
                                    break
                        break
                    if action is None:
                        break
                    stat = execute_action(action)
                    if stat == -1:
                        return action_trajectory
                occupancy_map, key_info = self.generate_occupancy_map()

        self.render_mode = prev_render_mode
        return action_trajectory

    def get_action(self, next_pos, random_action_prob=0.0):
        """Get the corresponding action for the next position"""
        agent_pos = self.agent_pos
        agent_dir = self.agent_dir
        next_type = self.grid.get(*next_pos)
        action = None
        p = random.random()
        action_list = [Actions.left, Actions.right, Actions.forward]
        if p < random_action_prob:
            return random.choice(action_list), True
        if not isinstance(next_type, Wall):
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
        return action, False


def minigrid_plan(env: BaseMiniGridEnv, goal_pos: tuple, random_action_prob: float = 0.1):
    """Planning method for minigrid"""
    test_env = BaseMiniGridEnv(render_mode="none")
    test_env.reset(goal_poses=[
                   goal_pos], agent_start_pos=env.agent_pos, agent_start_dir=env.agent_dir)
    test_env.grid = deepcopy(env.grid)
    action_trajectory = test_env.optimal_action_trajectory(
        goal_pos, random_action_prob)
    return action_trajectory


if __name__ == "__main__":
    start_pos = (2, 2)
    goal_pos = (8, 8)

    env = BaseMiniGridEnv(render_mode="human", agent_start_pos=start_pos)
    env.reset()

    start_pos = env.agent_pos
    action_trajectory = minigrid_plan(env, goal_pos, 0.1)
    for action in action_trajectory:
        obs, reward, terminated, truncated, info = env.step(action)
