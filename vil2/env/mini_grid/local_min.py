from __future__ import annotations
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.actions import Actions
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from vil2.env.mini_grid.base import BaseMiniGridEnv, minigrid_plan


class LocalMinGridEnv(BaseMiniGridEnv):
    """MiniGrid with local minimum setting"""

    def __init__(self,
                 grid_size=10,
                 agent_start_pos=(1, 1),
                 agent_start_dir=0,
                 max_steps: int | None = None,
                 **kwargs):
        super().__init__(
            grid_size=grid_size,
            max_steps=max_steps,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the goal"

    def _gen_grid(self, width, height):
        """Gnerate everything in the scene"""
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square
        for goal_pos in self.goal_poses:
            self.put_obj(Goal(), goal_pos[0], goal_pos[1])

        # Place blocks inside
        l_width_idx = int(0.3 * self.width)
        h_width_idx = int(0.7 * self.width)
        l_height_idx = int(0.3 * self.height)
        h_height_idx = int(0.7 * self.height)
        for i in range(l_width_idx, h_width_idx):
            for j in range(l_height_idx, h_height_idx):
                self.put_obj(Wall(), i, j)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"


if __name__ == "__main__":
    start_pos = (2, 2)
    goal_pos = (8, 2)
    way_points = [(2, 8), (8, 8), (8, 2)]

    env = LocalMinGridEnv(render_mode="human", agent_start_pos=start_pos)
    env.reset(goal_poses=[goal_pos])

    for way_point in way_points:
        start_pos = env.agent_pos
        print(f"start_pos: {start_pos}, way_point: {way_point}")
        action_trajectory = minigrid_plan(env, way_point, 0.1)
        for action in action_trajectory:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"pos: {env.agent_pos}, action: {Actions(action).name}")
