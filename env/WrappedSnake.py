import sys
import os
sys.path.append(os.path.dirname(__file__))
from snakes import SnakeEatBeans
import numpy as np
from multiagentenv import MultiAgentEnv
from submissions.blue import policy as blue_policy

class SnakeEnv(MultiAgentEnv):
    """
    引用自smac库的mutltiagent
    """
    def __init__(self,config:dict,render = True):
        self.snake_env = SnakeEatBeans(config)
        self.episode_limit = self.snake_env.max_step
        self.n_agents = self.snake_env.n_player
        self.n_controlled_player = 3
        self.sight_range = self.snake_env.sight_range
        self.n_beans = self.snake_env.n_beans
        self.obs_dim = self.sight_range * self.sight_range * 4 + self.n_beans * 2 + 2 + self.n_controlled_player * 2
        self.current_obs = None
        self.action_index_to_value = np.eye(4)
        self.action_value_to_index = {-2:0,-1:1,1:2,2:3}
        self.render = config['render']
        self.max_len_ratio  = config['rewards']['max_len_ratio']

    def get_env_info(self):
        return {
            "state_shape": 3 * self.obs_dim,
            "obs_shape": self.obs_dim,
            "n_actions": 4,
            "n_agents": self.n_controlled_player,
            "episode_limit": self.snake_env.max_step,
        }

    def reset(self):
        self.snake_env.reset(render=self.render)

    def get_state(self):
        return np.array(self.get_obs()[:3]).reshape(-1)
        # obs = self.snake_env.get_all_observes()

    #
    def get_avail_actions(self):
        """
        能动的方向值为1,不能动的方向值为0
        """
        self.current_obs = self.snake_env.get_all_observes()
        avail_action = []
        for i in range(3):
            avail_action.append(self.get_avail_agent_actions(i))
        return avail_action

    def get_avail_agent_actions(self, agent_id):
        avail_direction = [1,1,1,1]#按顺序排-2,-1,1,2，对应
        last_dir = self.current_obs[agent_id]['last_direction']
        if last_dir is None:
            return avail_direction
        avail_direction[self.action_value_to_index[last_dir]] = 0
        return avail_direction

    def close(self):
        return

    def get_obs(self):
        obs = self.snake_env.get_all_observes()
        beans_pos = np.array(obs[0][1]).reshape(-1)
        board_width = obs[0]['board_width']
        board_height = obs[0]['board_height']
        new_obs_list = []

        for i in range(self.n_controlled_player):
            single_obs = obs[i]
            snake_head = single_obs[i + 2][0]
            snake_grid = np.ones([self.sight_range * 2,self.sight_range * 2]) * -1
            for j in range(self.n_agents):
                for coordinate in single_obs[j + 2]:
                    if pow(coordinate[0] - snake_head[0],2) + pow(coordinate[1] - snake_head[1],2) <= self.sight_range * self.sight_range:
                        snake_grid[coordinate[0] - snake_head[0]][coordinate[1] - snake_head[1]] = j
            modified_obs = np.concatenate([snake_grid.flatten(),beans_pos])
            modified_obs = np.concatenate([modified_obs, [board_width, board_height]])
            for j in range(3):#append allies' head
                modified_obs = np.concatenate([modified_obs,single_obs[j + 2][0]])
            new_obs_list.append(modified_obs)
        return new_obs_list

    def step(self, actions):
        red_action = [[self.action_index_to_value[actions[i]].tolist()] for i in range(3)]
        blue_action = blue_policy(self.current_obs[3:])
        all_action = red_action + blue_action
        obs, rewards, terminal, info = self.snake_env.step(all_action)
        #把关于死亡惩罚和吃豆子奖励写在snake-env里了，所以这里不用修改
        snake_length = [len(self.snake_env.players[i].segments) for i in range(3)]
        max_length = np.max(snake_length)
        all_reward = np.sum(rewards[:3]) + self.max_len_ratio * (max_length - 3)
        #由于info should be int type
        info_int = {"total_red_score":np.sum(info["score"][:3]),"total_blue_score":np.sum(info["score"][3:]),"max_len_red":np.max(snake_length)}

        return all_reward,terminal,info_int
