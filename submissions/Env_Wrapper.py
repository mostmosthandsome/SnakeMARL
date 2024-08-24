import numpy as np

obs_dim = 148

def wrap_obs(obs):
    """
    used to wrap the obs from origin
    """
    beans_pos = np.array(obs[0][1])
    board_param = np.array([obs[0]['board_height'],obs[0]['board_width']])
    new_obs_list = []
    for i in range(3):
        single_obs = obs[i]
        snake_head = np.array(single_obs[i + 2][0])
        beans_pos_reference_to_snake = np.stack([
            (beans_pos[:,0] - snake_head[0] + board_param[0]) % board_param[0],
            (beans_pos[:,1] - snake_head[1] + board_param[1]) % board_param[1],
        ],axis=1)
        modified_obs =np.concatenate([beans_pos_reference_to_snake.reshape(-1),board_param])#22
        snake_sequence = np.ones([6,10,2]) * 100#120
        for j in range(6):
            for i in range(min(len(single_obs[j +2]),10)):
                #这个对应的是蛇向下和向右距离对应蛇的距离
                snake_sequence[j][i][0] = (single_obs[j+2][i][0] - snake_head[0] + board_param[0]) % board_param[0]
                snake_sequence[j][i][1] = (single_obs[j+2][i][1] - snake_head[1] + board_param[1]) % board_param[1]
        modified_obs = np.concatenate([modified_obs,snake_sequence.reshape(-1)])#142
        for j in range(3):  # append allies' head
            ally_head_pos = np.array([(single_obs[j + 2][0][0] - snake_head[0] + board_param[0]) % board_param[0],(single_obs[j + 2][0][1] - snake_head[1] + board_param[1])% board_param[1]])
            modified_obs = np.concatenate([modified_obs, ally_head_pos.reshape(-1)])
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