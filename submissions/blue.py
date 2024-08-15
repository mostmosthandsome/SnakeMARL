import numpy as np

act_dim = 4
num = 3

def policy(observation):

    eye = np.eye(act_dim)
    rand_acts = np.random.randint(act_dim, size=num)
    acts = []
    for act in rand_acts:
        acts.append([eye[act].tolist()])


    return acts