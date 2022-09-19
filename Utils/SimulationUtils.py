import sys
sys.path.append('./')
sys.path.append('../')
import numpy as np
import copy
import torch

from Utils.ComputationUtils import makeChoice



def to_255(torch_img_CHW):
    numpy_img=torch_img_CHW.numpy().transpose(1,2,0)
    normalized_array = (numpy_img - numpy_img.min())/(numpy_img.max() - numpy_img.min()) # this set the range from 0 till 1
    img_array = (normalized_array * 255).astype(np.uint8) # set is to a range from 0 till 255
    return img_array


def find_dist(a, b):
    # return np.NaN
    if a==b:
        return 0.0
    global dij_distance_df
    return dij_distance_df.loc[(dij_distance_df['pos1']==str(a)) & (dij_distance_df['pos2']==str(b))].iloc[0]['dis']


def to_tile(x):
    return int(np.floor(x/25)+1)


def to_pixel(x):
    return 25*(x-1)


def get_action(agent):
    action_list={"up":1,"down":2,"left":3,"right":4}
    estimated, agent_estimation, available_dir_index = agent.estimateDir()
    # Deal with the case when utilities are all negative
    copy_estimated = copy.deepcopy(estimated)
    copy_estimated[available_dir_index] = copy_estimated[available_dir_index] - np.min(copy_estimated[available_dir_index]) + 1
    action_name = agent.dir_list[makeChoice(copy_estimated)]
    return action_list[action_name], action_name, estimated, agent_estimation


def isSameState(state1, state2, compare_pacmanStateOnly = False):
    if compare_pacmanStateOnly:
        return str(state1['pacman_pos'])==str(state2['pacman_pos'])

    for k in state1.keys():
        if not str(state1[k])==str(state2[k]):
            return False
    return True


def get_proper_length(x):
    if len(x.shape)==1:
        return 1
    elif len(x.shape)==2:
        return x.shape[1]
    else:
        raise NotImplementedError


def atCorner(available_dir):
    dir_str = "".join(available_dir)
    if len(available_dir) != 2:
        return False
    elif dir_str == "leftup" or dir_str == "leftdown" or dir_str == "rightup" or dir_str == "rightdown":
        return True
    else:
        return False

# -------------------------------------------------

def proper_squeeze(x):
    if len(x.shape)==1:
        return torch.unsqueeze(x, 1)
    else:
        return x


def ghost_state_filter(x):
    if x == 0:
        return 1
    return x


def ghost_pos_filter(x):
    tile_pos=tuple(map(to_tile, x))
    if to_tile(475) == tile_pos[1]:
        print("Bottom pixel of home!")
    if 11<=tile_pos[0]<=18 and tile_pos[1]==20:
        print("Ghost reaches at the bottom of home!")
        return (tile_pos[0],19)
        # return tile_pos
    else:
        return tile_pos


def convert_state(obs):
    # Record game status
    re={}
    re['pacman_pos'] = (to_tile(obs['pacman_pos'][0]), to_tile(obs['pacman_pos'][1]))
    obs['energizer_locations']=proper_squeeze(obs['energizer_locations'])
    re['energizer_pos'] = [tuple(map(to_tile,obs['energizer_locations'][:,i].tolist())) for i in range(obs['energizer_locations'].shape[1])]
    obs['reward_locations']=proper_squeeze(obs['reward_locations'])
    re['bean_pos'] = [tuple(map(to_tile,obs['reward_locations'][:,i].tolist())) for i in range(obs['reward_locations'].shape[1])]
    re['Ghost_pos'] = [ghost_pos_filter(obs['ghost1_pos']), ghost_pos_filter(obs['ghost2_pos'])]
    re['fruit_type'] = int(obs['fruit_type'] + 2) if isinstance(obs['fruit_type'], float) else np.NaN # TODO [Jiaqi]: why +2 here
    re['fruit_pos'] = tuple(map(to_tile,obs['fruit_locations'].tolist())) if isinstance(obs['fruit_type'], float) else np.NaN
    re['Ghost_status'] = np.array([ghost_state_filter(obs['ghost1_state']),
                                ghost_state_filter(obs['ghost2_state'])], dtype=np.int)
    return re


def convert_Q(re, agent_Q):
    re["global_Q"] = agent_Q[:, 0]
    re["local_Q"] = agent_Q[:, 1]
    # re["optimistic_Q"] = agent_Q[:, 2]
    re["pessimistic_blinky_Q"] = agent_Q[:, 2]
    re["pessimistic_clyde_Q"] = agent_Q[:, 3]
    re["suicide_Q"] = agent_Q[:, 4]
    re["planned_hunting_Q"] = agent_Q[:, 5]
    return re


def diary(state, action_state):
    global diary_data, run_date, trial_count, step_count, tile_step_count, global_step
    diary_data['date'].append(run_date)
    diary_data['global_step'].append(global_step)
    diary_data['trialid'].append("{}".format(trial_count))
    diary_data["tile_step"].append(tile_step_count)
    diary_data['time_step'].append(step_count)
    diary_data['pacmanPos'].append(state['pacman_pos'])
    diary_data['energizers'].append(state['energizer_pos'])
    diary_data['beans'].append(state['bean_pos'])
    # diary_data['ghost1_distance'].append(state['Ghost_dist'][0])
    # diary_data['ghost2_distance'].append(state['Ghost_dist'][1])
    diary_data['ghost1_status'].append(state['Ghost_status'][0])
    diary_data['ghost2_status'].append(state['Ghost_status'][1])
    diary_data['ghost1_pos'].append(state['Ghost_pos'][0])
    diary_data['ghost2_pos'].append(state['Ghost_pos'][1])
    diary_data['fruit_pos'].append(state['fruit_pos'])
    diary_data['fruit_type'].append(state['fruit_type'])
    diary_data['possible_dir'].append(action_state[0])
    diary_data['pacman_dir'].append(action_state[1])
    # diary_data['time_spent'].append(time_spend)
    if "global_Q" not in diary_data:
        diary_data["global_Q"] = []
    if "global_Q" in state:
        diary_data["global_Q"].append(state['global_Q'])
    else:
        diary_data["global_Q"].append(np.nan)
    if "local_Q" not in diary_data:
        diary_data["local_Q"] = []
    if "local_Q" in state:
        diary_data["local_Q"].append(state['local_Q'])
    else:
        diary_data["local_Q"].append(np.nan)
    # if "optimistic_Q" not in diary_data:
    #     diary_data["optimistic_Q"] = []
    # if "optimistic_Q" in state:
    #     diary_data["optimistic_Q"].append(state['optimistic_Q'])
    # else:
    #     diary_data["optimistic_Q"].append(np.nan)
    if "pessimistic_blinky_Q" not in diary_data:
        diary_data["pessimistic_blinky_Q"] = []
    if "pessimistic_blinky_Q" in state:
        diary_data["pessimistic_blinky_Q"].append(state['pessimistic_blinky_Q'])
    else:
        diary_data["pessimistic_blinky_Q"].append(np.nan)

    if "pessimistic_clyde_Q" not in diary_data:
        diary_data["pessimistic_clyde_Q"] = []
    if "pessimistic_clyde_Q" in state:
        diary_data["pessimistic_clyde_Q"].append(state['pessimistic_clyde_Q'])
    else:
        diary_data["pessimistic_clyde_Q"].append(np.nan)

    if "suicide_Q" not in diary_data:
        diary_data["suicide_Q"] = []
    if "suicide_Q" in state:
        diary_data["suicide_Q"].append(state['suicide_Q'])
    else:
        diary_data["suicide_Q"].append(np.nan)

    if "planned_hunting_Q" not in diary_data:
        diary_data["planned_hunting_Q"] = []
    if "planned_hunting_Q" in state:
        diary_data["planned_hunting_Q"].append(state['planned_hunting_Q'])
    else:
        diary_data["planned_hunting_Q"].append(np.nan)

# ----------------------------

def get_proper_shape(t):
    if len(t.shape)>1:
        return t
    else:
        return t.view(-1,1)


def tile2index(tile_pos):
    return str(tile_pos[0]+(tile_pos[1]-1)*28)


def _pos2String(pos):
    return str((pos[1]-1)*28 + pos[0])

