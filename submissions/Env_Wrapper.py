import numpy as np

def wrap_obs(obs):
    """
    used to wrap the obs from origin
    """
    beans_pos = np.array(obs[0][1]).reshape(-1)
    board_width = obs[0]['board_width']
    board_height = obs[0]['board_height']
    new_obs_list = []
    for i in range(3):
        single_obs = obs[i]
        snake_head = single_obs[i + 2][0]
        snake_grid = np.ones([30, 30]) * -1
        for j in range(6):
            for coordinate in single_obs[j + 2]:
                if pow(coordinate[0] - snake_head[0], 2) + pow(coordinate[1] - snake_head[1],
                                                               2) <= 225:
                    snake_grid[coordinate[0] - snake_head[0]][coordinate[1] - snake_head[1]] = j
        modified_obs = np.concatenate([snake_grid.flatten(), beans_pos])
        modified_obs = np.concatenate([modified_obs, [board_width, board_height]])
        for j in range(3):  # append allies' head
            modified_obs = np.concatenate([modified_obs, single_obs[j + 2][0]])
        new_obs_list.append(modified_obs)
    return np.array(new_obs_list)

def get_available_agent_actions(obs, agent_id):
    action_value_to_index = {-2: 0, -1: 1, 1: 2, 2: 3}#将观测中的动作转化成0,1,2,3
    avail_direction = [1,1,1,1]#按顺序排-2,-1,1,2，对应
    last_dir = obs[agent_id]['last_direction']
    if last_dir is None:
        return avail_direction
    avail_direction[action_value_to_index[last_dir]] = 0
    return avail_direction