#!/usr/bin/env python3

# diffusion policy import
import torch
import os
import time
import numpy as np
import argparse
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import DPMSolverMultistepScheduler

# torch import
import torch as th
from torch.utils.data import TensorDataset

# gnn import
from torch_geometric.data import HeteroData

# visualization import
import matplotlib.pyplot as plt
from compression.utils.other import ObservationManager, ActionManager, getZeroState

# calculate loss
from scipy.optimize import linear_sum_assignment

# network utils import
from network_utils import ConditionalUnet1D

def str2bool(v):
    """
    This function converts string to boolean (mainly used for argparse)
    """

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def network_init(action_dim, obs_dim, obs_horizon, num_trajs, num_diffusion_iters, dataset_training, use_gnn, device):

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon,
        gnn_data=dataset_training[0] if use_gnn else None,
        use_gnn=use_gnn,
    )

    # example inputs
    noised_action = torch.randn((1, num_trajs, action_dim)).to(device)
    obs = torch.zeros((1, obs_horizon, obs_dim)).to(device)

    # example diffusion iteration
    diffusion_iter = torch.zeros((1,)).to(device)

    # the noise prediction network
    # takes noisy action, diffusion iteration and observation as input
    # predicts the noise added to action
    # you need to run noise_pred_net to initialize the network https://stackoverflow.com/questions/75550160/how-to-set-requires-grad-to-false-freeze-pytorch-lazy-layers
    x_dict = dataset_training[0].x_dict if use_gnn else None
    edge_index_dict = dataset_training[0].edge_index_dict if use_gnn else None
    noise = noise_pred_net(
        sample=noised_action,
        timestep=diffusion_iter,
        global_cond=obs.flatten(start_dim=1),
        x_dict=x_dict,
        edge_index_dict=edge_index_dict)

    # for this demo, we use DDPMScheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # try DPMSolverMultistepScheduler (not tested yet)
    # noise_scheduler = DPMSolverMultistepScheduler(
    #     num_train_timesteps=num_diffusion_iters,  
    #     beta_schedule='squaredcos_cap_v2',
    #     # clip_sample=True,
    #     prediction_type='epsilon'
    # )

    # device transfer
    _ = noise_pred_net.to(device)

    return noise_pred_net, noise_scheduler


def create_pair_obs_act(dirs, device, **kwargs):

    """
    Create dataset from npz files in dirs
    
    @param dirs: directory containing npz files
    @param device: device to transfer data to
    @param: kwargs: max_num_training_demos, percentage_training, percentage_eval, percentage_test
    @return dataset: TensorDataset
    """

    # unpack kwargs
    max_num_training_demos = kwargs.get('max_num_training_demos')
    percentage_training = kwargs.get('percentage_training')
    percentage_eval = kwargs.get('percentage_eval')
    percentage_test = kwargs.get('percentage_test')
    total_num_demos = max_num_training_demos / percentage_training

    # network type
    network_type = kwargs.get('network_type')

    # loop over dirs
    obs_data = th.tensor([]).to(device)
    traj_data = th.tensor([]).to(device)

    # list dirs in dirs
    dirs = [ f.path for f in os.scandir(dirs) if f.is_dir() ]

    idx = 0
    for dir in dirs:

        files = os.listdir(dir)
        files = [dir + '/' + file for file in files if file.endswith('.npz')]

        for file in files:
            
            data = np.load(file) # load data
            obs = data['obs'][:-1] # remove the last observation (since it is the observation of the goal state)
            acts = data['acts'] # actions

            # append to the data
            obs_data = th.cat((obs_data, th.from_numpy(obs).float().to(device)), 0)
            traj_data = th.cat((traj_data, th.from_numpy(acts).float().to(device)), 0)
            idx += 1
            if idx >= total_num_demos:
                break
        else:
            continue
        break

    # print out the total data size
    # print(f"original data size: {obs_data.shape[0] * traj_data.shape[1]}") # obs_data.shape[0] is the num of demos, traj_data.shape[1] is num of trajs per each demo (default is 10)
    print(f"original data size: {obs_data.shape[0]}") # obs_data.shape[0] is the num of demos, traj_data.shape[1] is num of trajs per each demo (default is 10)

    # rearanage the data for diffusion model
    if network_type == 'diffusion':
        assert obs_data.shape[0] == traj_data.shape[0], "obs_data and traj_data must have the same number of samples"
        dataset_obs = th.tensor([]).to(device)
        dataset_acts = th.tensor([]).to(device)
        idx = 0
        for i in range(obs_data.shape[0]): # loop over samples
            for j in range(traj_data.shape[1]): # loop over expert demonstrations (10)
                dataset_obs = th.cat((dataset_obs, obs_data[i, 0, :].unsqueeze(0).unsqueeze(0)), 0)
                dataset_acts = th.cat((dataset_acts, traj_data[i, j, :].unsqueeze(0).unsqueeze(0)), 0)
                idx += 1
                if idx >= total_num_demos:
                    break
            else:
                continue
            break
    else:
        dataset_obs = obs_data
        dataset_acts = traj_data

    # split the dataset into training, eval, and test
    dataset_obs_training = dataset_obs[0:int(dataset_obs.shape[0]*percentage_training)]
    dataset_acts_training = dataset_acts[0:int(dataset_acts.shape[0]*percentage_training)]
    dataset_obs_eval = dataset_obs[int(dataset_obs.shape[0]*percentage_training):int(dataset_obs.shape[0]*(percentage_training+percentage_eval))]
    dataset_acts_eval = dataset_acts[int(dataset_acts.shape[0]*percentage_training):int(dataset_acts.shape[0]*(percentage_training+percentage_eval))]
    dataset_obs_test = dataset_obs[int(dataset_obs.shape[0]*(percentage_training+percentage_eval)):]
    dataset_acts_test = dataset_acts[int(dataset_acts.shape[0]*(percentage_training+percentage_eval)):]

    # resize the dataset
    dataset_obs_training = dataset_obs_training.squeeze(1)
    dataset_acts_training = dataset_acts_training.squeeze(1)
    dataset_obs_eval = dataset_obs_eval.squeeze(1)
    dataset_acts_eval = dataset_acts_eval.squeeze(1)
    dataset_obs_test = dataset_obs_test.squeeze(1)
    dataset_acts_test = dataset_acts_test.squeeze(1)

    # print out the dataset size
    print(f"dataset after rearranging: training data size: {dataset_obs_training.shape[0]}")

    return dataset_obs_training, dataset_acts_training, dataset_obs_eval, dataset_acts_eval, dataset_obs_test, dataset_acts_test

def create_gnn_dataset_from_obs_and_acts(dataset_obs, dataset_acts, device):

    """
    This function generates a dataset for GNN
    """

    dataset = []

    assert dataset_obs.shape[0] == dataset_acts.shape[0], "the length of dataset_obs and dataset_acts should be the same"

    for i in range(dataset_obs.shape[0]):

        " ********************* GET NODES ********************* "

        # nodes you need for GNN
        # dataset = [f_v, f_a, yaw_dot, f_g,  [f_ctrl_pts_o0], bbox_o0, [f_ctrl_pts_o1], bbox_o1 ,...]
        # dim         3    3      1      3          30            3          30              3 
        # 0. current state 
        # 1. goal state 
        # 2. observation
        # In the current setting f_obs is a realative state from the current state so we pass f_v, f_z, yaw_dot to the current state node

        " ********************* MAKE A DATA OBJECT FOR HETEROGENEUS GRAPH ********************* "

        data = HeteroData()

        # add nodes
        # get num of obst
        num_of_obst = int(len(dataset_obs[i][10:])/33)

        # if type(dataset_obs[i]) is np.ndarray:
        #     warnings.warn("f_obs_n is a numpy array - converting it to a torch tensor")
        #     dataset_obs[i] = th.tensor(dataset_obs[i], dtype=th.double).to(device)

        feature_vector_for_current_state = dataset_obs[i][0:7]
        feature_vector_for_goal = dataset_obs[i][7:10]
        feature_vector_for_obs = dataset_obs[i][10:]

        dist_current_state_goal = np.linalg.norm(feature_vector_for_goal[:3].to('cpu').numpy())

        dist_current_state_obs = []
        dist_goal_obs = []
        for j in range(num_of_obst):
            dist_current_state_obs.append(np.linalg.norm(feature_vector_for_obs[33*j:33*j+3].to('cpu').numpy()))
            dist_goal_obs.append(np.linalg.norm((feature_vector_for_goal[:3] - feature_vector_for_obs[33*j:33*j+3]).to('cpu').numpy()))

        dist_obst_to_obst = []
        for j in range(num_of_obst):
            for k in range(num_of_obst):
                if j != k:
                    dist_obst_to_obst.append((np.linalg.norm((feature_vector_for_obs[33*j:33*j+3] - feature_vector_for_obs[33*k:33*k+3]).to('cpu').numpy())))


        " ********************* MAKE A DATA OBJECT FOR HETEROGENEUS GRAPH ********************* "

        # add nodes
        data["current_state"].x = feature_vector_for_current_state.unsqueeze(0).float().to(device)
        data["goal_state"].x = feature_vector_for_goal.unsqueeze(0).float().to(device)
        data["observation"].x = th.stack([feature_vector_for_obs[33*j:33*(j+1)] for j in range(num_of_obst)], dim=0).float().to(device)

        # add edges
        if num_of_obst == 2:
            data["current_state", "dist_current_state_to_goal_state", "goal_state"].edge_index = th.tensor([
                                                                                        [0],  # idx of source nodes (current state)
                                                                                        [0],  # idx of target nodes (goal state)
                                                                                        ],dtype=th.int64)
            data["current_state", "dist_current_state_to_observation", "observation"].edge_index = th.tensor([
                                                                                            [0],  # idx of source nodes (current state)
                                                                                            [0],  # idx of target nodes (observation)
                                                                                            ],dtype=th.int64)
            data["observation", "dist_obs_to_goal", "goal_state"].edge_index = th.tensor([
                                                                                            [0, 0],  # idx of source nodes (observation)
                                                                                            [0, 1],  # idx of target nodes (goal state)
                                                                                            ],dtype=th.int64)
            data["observation", "dist_observation_to_current_state", "current_state"].edge_index = th.tensor([
                                                                                            [0, 0],  # idx of source nodes (observation)
                                                                                            [0, 1],  # idx of target nodes (current state)
                                                                                            ],dtype=th.int64)
            data["goal_state", "dist_goal_state_to_current_state", "current_state"].edge_index = th.tensor([
                                                                                            [0],  # idx of source nodes (goal state)
                                                                                            [0],  # idx of target nodes (current state)
                                                                                            ],dtype=th.int64)
            data["goal_state", "dist_to_obs", "observation"].edge_index = th.tensor([
                                                                                            [0, 0],  # idx of source nodes (goal state)
                                                                                            [0, 1],  # idx of target nodes (observation)
                                                                                            ],dtype=th.int64)

        elif num_of_obst == 1:

            data["current_state", "dist_current_state_to_goal_state", "goal_state"].edge_index = th.tensor([
                                                                                        [0],  # idx of source nodes (current state)
                                                                                        [0],  # idx of target nodes (goal state)
                                                                                        ],dtype=th.int64)
            data["current_state", "dist_current_state_to_observation", "observation"].edge_index = th.tensor([
                                                                                            [0],  # idx of source nodes (current state)
                                                                                            [0],  # idx of target nodes (observation)
                                                                                            ],dtype=th.int64)
            data["observation", "dist_obs_to_goal", "goal_state"].edge_index = th.tensor([
                                                                                            [0],  # idx of source nodes (observation)
                                                                                            [0],  # idx of target nodes (goal state)
                                                                                            ],dtype=th.int64)
            data["observation", "dist_observation_to_current_state", "current_state"].edge_index = th.tensor([
                                                                                            [0],  # idx of source nodes (observation)
                                                                                            [0],  # idx of target nodes (current state)
                                                                                            ],dtype=th.int64)
            data["goal_state", "dist_goal_state_to_current_state", "current_state"].edge_index = th.tensor([
                                                                                            [0],  # idx of source nodes (goal state)
                                                                                            [0],  # idx of target nodes (current state)
                                                                                            ],dtype=th.int64)
            data["goal_state", "dist_to_obs", "observation"].edge_index = th.tensor([
                                                                                            [0],  # idx of source nodes (goal state)
                                                                                            [0],  # idx of target nodes (observation)
                                                                                            ],dtype=th.int64)

        # add edge weights
        data["current_state", "dist_current_state_to_goal_state", "goal_state"].edge_attr = dist_current_state_goal
        data["current_state", "dist_current_state_to_observation", "observation"].edge_attr = dist_current_state_obs
        data["observation", "dist_obs_to_goal", "goal_state"].edge_attr = dist_goal_obs
        # make it undirected
        data["observation", "dist_observation_to_current_state", "current_state"].edge_attr = dist_current_state_obs
        data["goal_state", "dist_goal_state_to_current_state", "current_state"].edge_attr = dist_current_state_goal
        data["goal_state", "dist_goal_to_obs", "observation"].edge_attr = dist_goal_obs

        # add ground truth trajectory
        data.acts = dataset_acts[i].unsqueeze(0).float().to(device)
        
        # add observation
        data.obs = dataset_obs[i].unsqueeze(0).float().to(device)

        # convert the data to the device
        data = data.to(device)
        # append data to the dataset
        dataset.append(data)

    " ********************* RETURN ********************* "

    return dataset

def create_dataset(dirs, device, **kwargs):
    
    """
    Create dataset from npz files in dirs
    @param dirs: directory containing npz files
    @param device: device to transfer data to
    @return dataset: TensorDataset
    """
    
    # get obs and acts
    dataset_obs_training, dataset_acts_training, dataset_obs_eval, dataset_acts_eval, dataset_obs_test, dataset_acts_test = create_pair_obs_act(dirs, device, **kwargs)

    # create dataset
    dataset_training = TensorDataset(dataset_obs_training, dataset_acts_training)
    dataset_eval = TensorDataset(dataset_obs_eval, dataset_acts_eval)
    dataset_test = TensorDataset(dataset_obs_test, dataset_acts_test)

    return dataset_training, dataset_eval, dataset_test

def create_gnn_dataset(dirs, device, **kwargs):

    """
    This function generates a dataset for GNN
    """

    " ********************* GET DATA ********************* "

    # get obs and acts
    dataset_obs_training, dataset_acts_training, dataset_obs_eval, dataset_acts_eval, dataset_obs_test, dataset_acts_test = create_pair_obs_act(dirs, device, **kwargs)

    # create dataset
    dataset_training = create_gnn_dataset_from_obs_and_acts(dataset_obs_training, dataset_acts_training, device)
    dataset_eval = create_gnn_dataset_from_obs_and_acts(dataset_obs_eval, dataset_acts_eval, device)
    dataset_test = create_gnn_dataset_from_obs_and_acts(dataset_obs_test, dataset_acts_test, device)

    return dataset_training, dataset_eval, dataset_test

def get_nactions(noise_pred_net, noise_scheduler, dataset, num_trajs, num_diffusion_iters, action_dim, use_gnn, device, is_visualize=True, num_eval=None):

    """
    This function generates a predicted trajectory
    """

    # set model to evaluation mode
    noise_pred_net.eval()

    # set batch size to 1
    B = 1

    # num of data to load
    num_data_to_load = len(dataset) if num_eval is None else num_eval

    # loop over the dataset
    print("start denoising...")
    expert_actions, nactions, nobses, times = [], [], [], []
    # for dataset_idx in tqdm(range(num_data_to_load), desc="data idx"):
    for dataset_idx in range(num_data_to_load):

        # stack the last obs_horizon (2) number of observations
        nobs = dataset[dataset_idx].obs if use_gnn else dataset[dataset_idx][0].unsqueeze(0)
        expert_action = dataset[dataset_idx].acts if use_gnn else dataset[dataset_idx][1].unsqueeze(0)

        # infer action
        with torch.no_grad():

            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, num_trajs, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            # start timer
            start_time = time.time()

            # for k in tqdm(noise_scheduler.timesteps, desc="diffusion iter k"):
            for k in noise_scheduler.timesteps:
                # predict noise

                if use_gnn:
                    x_dict = dataset[dataset_idx].x_dict
                    edge_index_dict = dataset[dataset_idx].edge_index_dict
                    noise_pred = noise_pred_net(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond,
                        x_dict=x_dict,
                        edge_index_dict=edge_index_dict
                    )
                else:
                    noise_pred = noise_pred_net(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            # end timer
            end_time = time.time()
            times.append(end_time - start_time)

        # append to the list
        expert_actions.append(expert_action.cpu().numpy())
        nactions.append(naction.squeeze(0).cpu().numpy())
        nobses.append(nobs.cpu().numpy())

        if is_visualize:
            # print out the computation time
            print("computation time: ", np.mean(times))
            # visualize trajectory
            visualize_trajectory(expert_action.cpu().numpy(), naction.squeeze(0).cpu().numpy(), nobs, dataset_idx)

    if not is_visualize: # used get_nactions used in evaluation
        return expert_actions, nactions, nobses

def plot_obstacles(start_state, f_obs, ax):

    w_obs_poses = []
    p0 = start_state.w_pos[0] # careful here: we assume the agent pos is at the origin - as the agent moves we expect the obstacles shift accordingly
    
    # extract each obstacle trajs
    num_obst = int(len(f_obs[0][10:])/33)
    for i in range(num_obst):
        
        # get each obstacle's trajectory
        f_obs_each = f_obs[0][10+33*i:10+33*(i+1)]
    
        # get each obstacle's poses in that trajectory in the world frame
        for i in range(int(len(f_obs_each[:-3])/3)):
            w_obs_poses.append((f_obs_each[3*i:3*i+3] - p0.T).tolist())

        # get each obstacle's bbox
        bbox = f_obs_each[-3:]

        # plot the bbox
        for idx, w_obs_pos in enumerate(w_obs_poses):
            # obstacle's position
            # bbox (8 points)
            p1 = [w_obs_pos[0] + bbox[0]/2, w_obs_pos[1] + bbox[1]/2, w_obs_pos[2] + bbox[2]/2]
            p2 = [w_obs_pos[0] + bbox[0]/2, w_obs_pos[1] + bbox[1]/2, w_obs_pos[2] - bbox[2]/2]
            p3 = [w_obs_pos[0] + bbox[0]/2, w_obs_pos[1] - bbox[1]/2, w_obs_pos[2] + bbox[2]/2]
            p4 = [w_obs_pos[0] + bbox[0]/2, w_obs_pos[1] - bbox[1]/2, w_obs_pos[2] - bbox[2]/2]
            p5 = [w_obs_pos[0] - bbox[0]/2, w_obs_pos[1] + bbox[1]/2, w_obs_pos[2] + bbox[2]/2]
            p6 = [w_obs_pos[0] - bbox[0]/2, w_obs_pos[1] + bbox[1]/2, w_obs_pos[2] - bbox[2]/2]
            p7 = [w_obs_pos[0] - bbox[0]/2, w_obs_pos[1] - bbox[1]/2, w_obs_pos[2] + bbox[2]/2]
            p8 = [w_obs_pos[0] - bbox[0]/2, w_obs_pos[1] - bbox[1]/2, w_obs_pos[2] - bbox[2]/2]
            # bbox lines (12 lines)
            if idx == 0 and i == 0:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], lw=2, alpha=0.7, c='blue', label='Obstacle')
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], lw=2, alpha=0.7, c='blue')
            ax.plot([p1[0], p3[0]], [p1[1], p3[1]], [p1[2], p3[2]], lw=2, alpha=0.7, c='blue')
            ax.plot([p1[0], p5[0]], [p1[1], p5[1]], [p1[2], p5[2]], lw=2, alpha=0.7, c='blue')
            ax.plot([p2[0], p4[0]], [p2[1], p4[1]], [p2[2], p4[2]], lw=2, alpha=0.7, c='blue')
            ax.plot([p2[0], p6[0]], [p2[1], p6[1]], [p2[2], p6[2]], lw=2, alpha=0.7, c='blue')
            ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], lw=2, alpha=0.7, c='blue')
            ax.plot([p3[0], p7[0]], [p3[1], p7[1]], [p3[2], p7[2]], lw=2, alpha=0.7, c='blue')
            ax.plot([p4[0], p8[0]], [p4[1], p8[1]], [p4[2], p8[2]], lw=2, alpha=0.7, c='blue')
            ax.plot([p5[0], p6[0]], [p5[1], p6[1]], [p5[2], p6[2]], lw=2, alpha=0.7, c='blue')
            ax.plot([p5[0], p7[0]], [p5[1], p7[1]], [p5[2], p7[2]], lw=2, alpha=0.7, c='blue')
            ax.plot([p6[0], p8[0]], [p6[1], p8[1]], [p6[2], p8[2]], lw=2, alpha=0.7, c='blue')
            ax.plot([p7[0], p8[0]], [p7[1], p8[1]], [p7[2], p8[2]], lw=2, alpha=0.7, c='blue')

def plot_pos(actions, am, ax, label, num_vectors_pos, num_vectors_yaw, start_state):

    # color (expert: green, student: orange)
    color = 'green' if label == 'Expert' else 'orange'

    for action_idx in range(actions.shape[0]):

        action = actions[action_idx, :].reshape(1, -1)
        traj = am.denormalizeTraj(action)
        
        # convert the trajectory to a b-spline
        w_posBS, w_yawBS = am.f_trajAnd_w_State2wBS(traj, start_state)
        time_pos = np.linspace(w_posBS.getT0(), w_posBS.getTf(), num_vectors_pos)
        time_yaw = np.linspace(w_yawBS.getT0(), w_yawBS.getTf(), num_vectors_yaw)

        # plot the predicted trajectory
        if action_idx == 0 and label == 'Expert':
            # plot the start and goal position
            ax.scatter(w_posBS.pos_bs[0](w_posBS.getT0()), w_posBS.pos_bs[1](w_posBS.getT0()), w_posBS.pos_bs[2](w_posBS.getT0()), s=100, c='pink', marker='o', label='Start')
        else:
            label = None
        
        # plot trajectory
        ax.plot(w_posBS.pos_bs[0](time_pos), w_posBS.pos_bs[1](time_pos), w_posBS.pos_bs[2](time_pos), lw=4, alpha=0.7, label=label, c=color)
        # plot yaw direction
        ax.quiver(w_posBS.pos_bs[0](time_yaw), w_posBS.pos_bs[1](time_yaw), w_posBS.pos_bs[2](time_yaw), np.cos(w_yawBS.pos_bs[0](time_yaw)), np.sin(w_yawBS.pos_bs[0](time_yaw)), np.zeros_like(w_yawBS.pos_bs[0](time_yaw)), length=0.5, normalize=True, color='red')

        action_idx += 1
        if action_idx > len(actions):
            break

def plot_yaw(actions, am, ax, label, num_vectors_yaw, start_state):

    # color (expert: green, student: orange)
    color = 'green' if label == 'Expert' else 'orange'

    for action_idx in range(actions.shape[0]):

        action = actions[action_idx, :].reshape(1, -1)
        traj = am.denormalizeTraj(action)

        # convert the trajectory to a b-spline
        _, w_yawBS = am.f_trajAnd_w_State2wBS(traj, start_state)
        time_yaw = np.linspace(w_yawBS.getT0(), w_yawBS.getTf(), num_vectors_yaw)

        if not action_idx == 0:
            label = None
        else:
            ax.plot(time_yaw, w_yawBS.pos_bs[0](time_yaw), lw=4, alpha=0.7, label='Diffusion', c=color)

        action_idx += 1
        if action_idx > len(actions):
            break

def visualize_trajectory(expert_action, action_pred, nobs, image_idx):

    # interpolation parameters
    num_vectors_pos = 100
    num_vectors_yaw = 10

    # get start state
    start_state = getZeroState() # TODO (hardcoded)

    # get action and observation manager for normalization and denormalization
    am = ActionManager() # get action manager
    om = ObservationManager() # get observation manager

    # plot 
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')

    # plot pos trajectories
    plot_pos(expert_action, am, ax, 'Expert', num_vectors_pos, num_vectors_yaw, start_state)
    plot_pos(action_pred, am, ax, 'Diffusion', num_vectors_pos, num_vectors_yaw, start_state)

    # plot the goal
    f_obs = om.denormalizeObservation(nobs.to('cpu').numpy())
    ax.scatter(f_obs[0][7], f_obs[0][8], f_obs[0][9], s=100, c='red', marker='*', label='Goal')
    
    # plot the obstacles
    # get w pos of the obstacles
    plot_obstacles(start_state, f_obs, ax)

    ax.grid(True)
    ax.legend(loc='best')
    ax.set_xlim(-2, 15)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

    # plot the yaw trajectory
    ax = fig.add_subplot(212)

    # plot yaw trajectories
    plot_yaw(expert_action, am, ax, 'Expert', num_vectors_yaw, start_state)
    plot_yaw(action_pred, am, ax, 'Diffusion', num_vectors_yaw, start_state)

    ax.grid(True)
    ax.legend(loc='best')
    ax.set_xlabel('Time')
    ax.set_ylabel('Yaw')
    # fig.savefig(f'/media/jtorde/T7/gdp/pngs/image_{image_idx}.png')
    plt.show()

def calculate_deep_panther_loss(batch, policy, yaw_loss_weight, is_lstm=False):
    """Calculate the supervised learning loss used to train the behavioral clone.

    Args:
        obs: The observations seen by the expert. If this is a Tensor, then
            gradients are detached first before loss is calculated.
        acts: The actions taken by the expert. If this is a Tensor, then its
            gradients are detached first before loss is calculated.

    Returns:
        loss: The supervised learning loss for the behavioral clone to optimize.
        stats_dict: Statistics about the learning process to be logged.

    """

    # get the observation and action
    
    obs = batch[0]
    acts = batch[1]

    # (TODO hardcoded)
    traj_size_pos_ctrl_pts = 15
    traj_size_yaw_ctrl_pts = 6

    # set policy to train mode
    policy.train()

    # get the predicted action

    if is_lstm:
        # TODO: length of the sequence is hardcoded
        length_of_sequence = 1
        obs = th.reshape(obs, (obs.shape[0], length_of_sequence, obs.shape[1])) # (batch_size, sequence_length, hidden_size)
        pred_acts = policy(obs)
    else:

        print("obs shape: ", obs.shape)
        pred_acts = policy(obs)
        print("pred_acts shape: ", pred_acts.shape)

    # get size 
    num_of_traj_per_action=list(acts.shape)[1] #acts.shape is [batch size, num_traj_action, size_traj]
    batch_size=list(acts.shape)[0] #acts.shape is [batch size, num_of_traj_per_action, size_traj]

    # initialize the distance matrix
    distance_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action)
    distance_pos_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action) 
    distance_yaw_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action) 
    distance_time_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action)
    distance_pos_matrix_within_expert= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action)

    #Expert --> i
    #Student --> j
    for i in range(num_of_traj_per_action):
        for j in range(num_of_traj_per_action):

            expert_i=       acts[:,i,:].float(); #All the elements
            student_j=      pred_acts[:,j,:].float() #All the elements

            expert_pos_i=   acts[:,i,0:traj_size_pos_ctrl_pts].float()
            student_pos_j=  pred_acts[:,j,0:traj_size_pos_ctrl_pts].float()

            expert_yaw_i=   acts[:,i,traj_size_pos_ctrl_pts:(traj_size_pos_ctrl_pts+traj_size_yaw_ctrl_pts)].float()
            student_yaw_j=  pred_acts[:,j,traj_size_pos_ctrl_pts:(traj_size_pos_ctrl_pts+traj_size_yaw_ctrl_pts)].float()

            expert_time_i=       acts[:,i,-1:].float(); #Time. Note: Is you use only -1 (instead of -1:), then distance_time_matrix will have required_grad to false
            student_time_j=      pred_acts[:,j,-1:].float() #Time. Note: Is you use only -1 (instead of -1:), then distance_time_matrix will have required_grad to false

            distance_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_i, student_j), dim=1)
            distance_pos_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_pos_i, student_pos_j), dim=1)
            distance_yaw_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_yaw_i, student_yaw_j), dim=1)
            distance_time_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_time_i, student_time_j), dim=1)

            #This is simply to delete the trajs from the expert that are repeated
            expert_pos_j=   acts[:,j,0:traj_size_pos_ctrl_pts].float()
            distance_pos_matrix_within_expert[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_pos_i, expert_pos_j), dim=1)

    is_repeated=th.zeros(batch_size, num_of_traj_per_action, dtype=th.bool)

    for i in range(num_of_traj_per_action):
        for j in range(i+1, num_of_traj_per_action):
            is_repeated[:,j]=th.logical_or(is_repeated[:,j], th.lt(distance_pos_matrix_within_expert[:,i,j], 1e-7))

    assert distance_matrix.requires_grad==True
    assert distance_pos_matrix.requires_grad==True
    assert distance_yaw_matrix.requires_grad==True
    assert distance_time_matrix.requires_grad==True

    #Option 1: Solve assignment problem
    A_matrix=th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action)

    for index_batch in range(batch_size):         

        cost_matrix=distance_pos_matrix[index_batch,:,:]
        map2RealRows=np.array(range(num_of_traj_per_action))
        map2RealCols=np.array(range(num_of_traj_per_action))

        rows_to_delete=[]
        for i in range(num_of_traj_per_action): #for each row (expert traj)
            if(is_repeated[index_batch,i]==True): 
                rows_to_delete.append(i) #Delete that row

        cost_matrix=cost_matrix[is_repeated[index_batch,:]==False]   #np.delete(cost_matrix_numpy, rows_to_delete, axis=0)
        cost_matrix_numpy=cost_matrix.cpu().detach().numpy()

        # Solve assignment problem                                       
        row_indexes, col_indexes = linear_sum_assignment(cost_matrix_numpy)
        for row_index, col_index in zip(row_indexes, col_indexes):
            A_matrix[index_batch, map2RealRows[row_index], map2RealCols[col_index]]=1
            
    num_nonzero_A=th.count_nonzero(A_matrix); #This is the same as the number of distinct trajectories produced by the expert

    pos_loss=th.sum(A_matrix*distance_pos_matrix)/num_nonzero_A
    yaw_loss=th.sum(A_matrix*distance_yaw_matrix)/num_nonzero_A
    time_loss=th.sum(A_matrix*distance_time_matrix)/num_nonzero_A

    assert (distance_matrix.shape)[0]==batch_size, "Wrong shape!"
    assert (distance_matrix.shape)[1]==num_of_traj_per_action, "Wrong shape!"
    assert pos_loss.requires_grad==True
    assert yaw_loss.requires_grad==True
    assert time_loss.requires_grad==True

    loss = time_loss + pos_loss + yaw_loss_weight*yaw_loss

    return loss