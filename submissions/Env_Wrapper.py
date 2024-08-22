import numpy as np

obs_dim = 148

def wrap_obs(obs):
    """
    used to wrap the obs from origin
    """
    beans_pos = np.array(obs[0][1])
    board_param = np.array([obs[0]['board_width'],obs[0]['board_height']])
    new_obs_list = []
    for i in range(3):
        single_obs = obs[i]
        snake_head = np.array(single_obs[i + 2][0])

        modified_obs =np.concatenate([(beans_pos - snake_head).reshape(-1),board_param])#22
        snake_sequence = np.ones([6,10,2]) * 100#120
        for j in range(6):
            for i in range(min(len(single_obs[j +2]),10)):
                snake_sequence[j][i] = single_obs[j+2][i] - snake_head
        modified_obs = np.concatenate([modified_obs,snake_sequence.reshape(-1)])#142
        for j in range(3):  # append allies' head
            modified_obs = np.concatenate([modified_obs, single_obs[j + 2][0] - snake_head])
        new_obs_list.append(modified_obs)#148
    return np.array(new_obs_list)#450

def get_available_agent_actions(obs, agent_id):
    action_value_to_index = {'up': 0, 'down': 1, 'left': 2, 'right': 3}#将观测中的动作转化成0,1,2,3
    avail_direction = [1,1,1,1]#按顺序排-2,-1,1,2，对应
    last_dir = obs[agent_id]['last_direction']
    if last_dir is None:
        return avail_direction
    last_dir = last_dir[agent_id]
    avail_direction[action_value_to_index[last_dir]] = 0
    return avail_direction