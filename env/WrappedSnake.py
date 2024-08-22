import sys
import os
from os.path import dirname
ENV_DIR = dirname(__file__)
sys.path.append(dirname(ENV_DIR))
print(f"add {dirname(ENV_DIR)} to system path")
from env.snakes import SnakeEatBeans
import numpy as np
from env.multiagentenv import MultiAgentEnv
from submissions.blue import policy as blue_policy
from submissions.Env_Wrapper import wrap_obs,get_available_agent_actions,obs_dim

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
        self.obs_dim = obs_dim
        self.current_obs = None
        self.action_index_to_value = np.eye(4)
        self.action_value_to_index = {-2:0,-1:1,1:2,2:3}
        self.render = config['render']
        self.max_len_ratio  = config['rewards']['max_len_ratio']

    def get_env_info(self):
        return {
            "state_shape": 143,
            "obs_shape": self.obs_dim,
            "n_actions": 4,
            "n_agents": self.n_controlled_player,
            "episode_limit": self.snake_env.max_step,
        }

    def reset(self):

        self.snake_env.reset(render=self.render)

    def get_state(self):

        obs = self.snake_env.get_all_observes()
        snake_segments = [self.snake_env.players[i].segments for i in range(6)]
        snake_length = np.array([len(snake_segments[i]) for i in range(3)])#3
        beans_pos = np.array(obs[0][1]).reshape(-1)#20
        snake_sequence = np.ones([6, 10, 2]) * -1  # 120
        for j in range(6):
            for i in range(min(len(snake_segments[j]),10)):
                snake_sequence[j][i] = snake_segments[j][i]
        state = np.concatenate([snake_length,beans_pos,snake_sequence.reshape(-1)]) #143
        return state

    #
    def get_avail_actions(self):
        """
        能动的方向值为1,不能动的方向值为0
        """
        self.current_obs = self.snake_env.get_all_observes()
        avail_action = [self.get_avail_agent_actions(i) for i in range(3)]
        return avail_action

    def get_avail_agent_actions(self, agent_id):#把它写到controller里面去了，方便上传
        return get_available_agent_actions(self.current_obs,agent_id)

    def close(self):
        return

    def get_obs(self):
        obs = self.snake_env.get_all_observes(self.snake_env.before_info)
        return wrap_obs(obs)

    def step(self, actions):
        red_action = [[self.action_index_to_value[actions[i]].tolist()] for i in range(3)]
        blue_action = blue_policy(self.current_obs[3:])
        all_action = red_action + blue_action
        obs, rewards, terminal, info = self.snake_env.step(all_action)
        #把关于死亡惩罚和吃豆子奖励写在snake-env里了，所以这里不用修改

        snake_length = [len(self.snake_env.players[i].segments) for i in range(3)]
        max_length = np.max(snake_length)
        all_reward = np.sum(rewards[:3]) + self.max_len_ratio * (max_length - 3)
        snake_heads = obs[0]
        #由于info should be int type
        info_int = {"total_red_score":np.sum(info["score"][:3]),"total_blue_score":np.sum(info["score"][3:]),"max_len_red":np.max(snake_length)}

        return all_reward,terminal,info_int
