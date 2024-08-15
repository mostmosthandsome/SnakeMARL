import sys
import os
sys.path.append(os.path.dirname(__file__))
from snakes import SnakeEatBeans
import numpy as np

class SnakeEnv:
    def __init__(self,config:dict):
        self.snake_env = SnakeEatBeans(config)
        self.episode_limit = self.snake_env.max_step
        self.n_agents = self.snake_env.n_player


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
        return self.snake_env.get_all_observes()
