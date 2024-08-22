import numpy as np

obs_dim = 150

def wrap_obs(obs):
    """
    used to wrap the obs from origin
    """
    beans_pos = np.array(obs[0][1]).reshape(-1)
    board_param = np.array([obs[0]['board_width'],obs[0]['board_height']])
    new_obs_list = []
    for i in range(3):
        single_obs = obs[i]
        snake_head = np.array(single_obs[i + 2][0]).reshape(-1)
        modified_obs =np.concatenate([beans_pos,board_param,snake_head])#24
        snake_sequence = np.ones([6,10,2]) * -1#120
        for j in range(6):
            for i in range(min(len(single_obs[j +2]),10)):
                snake_sequence[j][i] = single_obs[j+2][i]
        modified_obs = np.concatenate([modified_obs,snake_sequence.reshape(-1)])#144
        for j in range(3):  # append allies' head
            modified_obs = np.concatenate([modified_obs, single_obs[j + 2][0]])
        new_obs_list.append(modified_obs)#150
    return np.array(new_obs_list)#450

def get_available_agent_actions(obs, agent_id):
    action_value_to_index = {-2: 0, -1: 1, 1: 2, 2: 3}#将观测中的动作转化成0,1,2,3
    avail_direction = [1,1,1,1]#按顺序排-2,-1,1,2，对应
    last_dir = obs[agent_id]['last_direction']
    if last_dir is None:
        return avail_direction
    avail_direction[action_value_to_index[last_dir]] = 0
    return avail_direction