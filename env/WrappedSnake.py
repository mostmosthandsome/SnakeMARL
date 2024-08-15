import sys
import os
sys.path.append(os.path.dirname(__file__))
from snakes import SnakeEatBeans

class SnakeEnv:
    def __init__(self,config:dict):
        self.snake_env = SnakeEatBeans(config)
        self.episode_limit = self.snake_env.max_step


    def get_env_info(self):
        return {
            "state_shape": 6 * (7 + 4),
            "obs_shape": 6 * (7 + 4),
            "n_actions": self.snake_env.n_player,
            "n_agents": self.snake_env.n_player,
            "episode_limit": self.snake_env.max_step,
        }