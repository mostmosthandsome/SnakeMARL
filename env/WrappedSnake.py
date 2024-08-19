import sys
import os
sys.path.append(os.path.dirname(__file__))
from snakes import SnakeEatBeans
import numpy as np
from multiagentenv import MultiAgentEnv
import torch as th

class SnakeEnv(MultiAgentEnv):
    """
    引用自
    """
    def __init__(self,config:dict):
        self.snake_env = SnakeEatBeans(config)
        self.episode_limit = self.snake_env.max_step
        self.n_agents = self.snake_env.n_player
        self.n_controlled_player = 3
        self.sight_range = self.snake_env.sight_range
        self.n_beans = self.snake_env.n_beans
        self.obs_dim = self.sight_range * self.sight_range * 4 + self.n_beans * 2 + 3


    def get_env_info(self):
        return {
            "state_shape": 6 * (7 + 4),
            "obs_shape": 6 * (7 + 4),
            "n_actions": self.snake_env.n_player,
            "n_agents": self.n_agents,
            "episode_limit": self.snake_env.max_step,
        }

    def reset(self):
        self.snake_env.reset(render=True)

    def get_state(self):
        return self.snake_env.get_all_observes()

    def get_avail_actions(self):
        return [[-2, 2, -1, 1]for i in range(self.n_agents)]

    def get_obs(self):
        obs = self.snake_env.get_all_observes()
        beans_pos = th.tensor(obs[0][1]).flatten(start_dim=0)
        board_width = th.tensor(obs[0]['board_width'])
        for i in range(self.n_controlled_player):
            single_obs = obs[i]
            snake_head = single_obs[i + 2][0]
            snake_grid = th.ones([self.sight_range * 2,self.sight_range * 2]) * -1
            for j in range(self.n_agents):
                for coordinate in single_obs[j + 2]:
                    if pow(coordinate[0] - snake_head[0],2) + pow(coordinate[1] - snake_head[1],2) <= self.sight_range * self.sight_range:
                        snake_grid[coordinate[0] - snake_head[0]][coordinate[1] - snake_head[1]] = j

        return th.cat([snake_grid,beans_pos,board_width],dim=1)

