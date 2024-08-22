from pymarl.src.modules.agents import RNNAgent
import numpy as np
import yaml
import os
import torch as th
from .Env_Wrapper import wrap_obs, get_available_agent_actions, obs_dim

model_path = "/home/handsome/study/course/year3.0/summer/snakes_v0/snakes/saved_result/models/qmix__2024-08-22_19-55-54/770000"



class AgentArgs:#这个类留着最后提交代码的时候改，就不用交default.yaml上去了
    def __init__(self,from_file=False):
        self.n_actions = 4
        if from_file:
            local_path = os.path.dirname(os.path.dirname(__file__))
            with open(os.path.join(local_path, "config", "default.yaml"), "r") as f:
                args = yaml.load(f)
            self.rnn_hidden_dim = args['rnn_hidden_dim']
            self.net_arch = args['net_arch']
            self.activate_fn = args['activate_fn']
            self.obs_last_action = args['obs_last_action']
            self.obs_agent_id = args['obs_agent_id']
            return
        self.rnn_hidden_dim = 32
        self.net_arch = [256,128,64]
        self.activate_fn = "ELU"
        self.obs_last_action = True
        self.obs_agent_id = True

class Controller:#基本上是抄的basic_controller
    def __init__(self,model_path):
        self.args = AgentArgs(from_file=True)#提交时改成False，并修改相应参数
        self.agent = RNNAgent(obs_dim + 7, self.args)
        self.agent.load_state_dict(th.load("{}/agent.th".format(model_path), map_location=lambda storage, loc: storage))
        self.init_hidden(1)
        self.last_action = None
        self.agent_index = th.tensor([[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]])
        self.action_index_to_value = np.eye(4)

    def init_hidden(self, batch_size):
        self.hidden_states = th.zeros(3, self.args.rnn_hidden_dim)  # (3,hidden_size)

    def load(self,path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def act(self,obs):#基本上抄自basic_controller的select_actions
        avail_actions = [get_available_agent_actions(obs,i) for i in range(3)]
        obs = th.tensor(wrap_obs(obs),dtype=th.float32)
        #抄自basic controller _build_inputs
        inputs = []
        inputs.append(obs)  # b1av
        if self.args.obs_last_action:
            if self.last_action is None:
                inputs.append(th.zeros([3,4]))
            else:
                inputs.append(self.last_action)
        if self.args.obs_agent_id:
            inputs.append(self.agent_index)
        obs = th.cat(inputs, dim=1)
        #改写一下basic_controller的forward函数
        last_hidden_state = self.hidden_states
        agent_outs, self.hidden_states = self.agent(obs, self.hidden_states)#输出(1,4)维的东西
        print("hidden state change:",(self.hidden_states - last_hidden_state).mean())
        print("agent_out= ",agent_outs)
        #抄自action_selectors的epsilon_greedy
        agent_outs[avail_actions == 0.0] = -float("inf")
        chosen_actions = agent_outs.max(dim=1)[1]
        one_hot_action = [[self.action_index_to_value[chosen_actions[i]].tolist()] for i in range(3)]
        return one_hot_action

agent = Controller(model_path)

def policy(obs):
    global agent
    return agent.act(obs)