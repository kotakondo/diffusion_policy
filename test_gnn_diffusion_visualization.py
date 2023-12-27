#!/usr/bin/env python3

# diffusion policy import
from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

# env import
import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from skvideo.io import vwrite
from IPython.display import Video
import gdown
import os
import time
# visualization import
import matplotlib.pyplot as plt
from compression.utils.other import ObservationManager, ActionManager, getZeroState

# Install data

import torch as th
from torch.utils.data import TensorDataset

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 3, 2, 1) # not sure why but the default kernel_size was set to 4

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
    
    #@markdown ### **Network Demo**

def main():

    # device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    device = 'cpu'

    # list npz files in the directory
    # dirs = "/home/jtorde/Research/puma_ws/src/puma/panther_compression/evals/tmp_dagger/2/demos/" #
    dirs = "/home/jtorde/Research/puma_ws/src/puma/panther_compression/evals-dir/evals4/tmp_dagger/2/demos/"
    # loop over dirs
    obs_data = th.tensor([]).to(device)
    traj_data = th.tensor([]).to(device)
    # list dirs in dirs
    dirs = subfolders = [ f.path for f in os.scandir(dirs) if f.is_dir() ]
    for dir in dirs:

        files = os.listdir(dir)
        files = [dir + '/' + file for file in files if file.endswith('.npz')]

        for idx, file in enumerate(files):
            
            data = np.load(file) # load data
            obs = data['obs'][:-1] # remove the last observation (since it is the observation of the goal state)
            acts = data['acts'] # actions

            # append to the data
            obs_data = th.cat((obs_data, th.tensor(obs).to(device)), 0)
            traj_data = th.cat((traj_data, th.tensor(acts).to(device)), 0)

    print("obs_data.shape: ", obs_data.shape)
    print("traj_data.shape: ", traj_data.shape)

    assert obs_data.shape[0] == traj_data.shape[0], "obs_data and traj_data must have the same number of samples"

    dataset_obs = th.tensor([]).to(device)
    dataset_acts = th.tensor([]).to(device)
    cnt = 0
    for i in range(obs_data.shape[0]): # loop over samples
        for j in range(traj_data.shape[1]): # loop over expert demonstrations (10)
            dataset_obs = th.cat((dataset_obs, obs_data[i, 0, :].unsqueeze(0).unsqueeze(0)), 0)
            dataset_acts = th.cat((dataset_acts, traj_data[i, j, :].unsqueeze(0).unsqueeze(0)), 0)
            cnt += 1
            if cnt > 100:
                break
    dataset_obs = dataset_obs.squeeze(1)
    dataset_acts = dataset_acts.squeeze(1)
    print("dataset_obs.shape: ", dataset_obs.shape)
    print("dataset_acts.shape: ", dataset_acts.shape)

    # create dataset
    dataset = TensorDataset(dataset_obs, dataset_acts)

    # parameters
    pred_horizon = 1 # changed Kota
    obs_horizon = 1 # changed Kota

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=1,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # observation and action dimensions
    # for now we don't use GNN
    obs_dim = 43
    action_dim = 22

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # example inputs
    noised_action = torch.randn((1, pred_horizon, action_dim))
    obs = torch.zeros((1, obs_horizon, obs_dim))

    diffusion_iter = torch.zeros((1,))

    # the noise prediction network
    # takes noisy action, diffusion iteration and observation as input
    # predicts the noise added to action
    noise = noise_pred_net(
        sample=noised_action,
        timestep=diffusion_iter,
        global_cond=obs.flatten(start_dim=1))

    # illustration of removing noise
    # the actual noise removal is performed by NoiseScheduler
    # and is dependent on the diffusion noise schedule
    denoised_action = noised_action - noise

    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    num_diffusion_iters = 100
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

    # device transfer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    _ = noise_pred_net.to(device)

    """ ********************* VISUALIZATION ********************* """

    # load model
    dir = "/home/jtorde/Research/diffusion_policy/models"
    # get the latest model in the directory
    files = os.listdir(dir)
    files = [dir + '/' + file for file in files if file.endswith('.pth')]
    files.sort(key=os.path.getmtime)
    model_path = files[-1]
    print("model_path: ", model_path)
    model = th.load(model_path, map_location=device)
    noise_pred_net.load_state_dict(model)

    # set model to evaluation mode
    noise_pred_net.eval()

    # set batch size to 1
    B = 1

    # number of trajectories for each demonstration (default: 10)
    num_trajs = 10

    # loop over the dataset
    for dataset_idx in range(20, dataset_obs.shape[0], num_trajs):
        expert_actions = []
        nactions = []
        times = []

        for traj_idx in range(num_trajs):

            # stack the last obs_horizon (2) number of observations
            test_obs = dataset_obs[dataset_idx+traj_idx].unsqueeze(0).numpy()
            expert_action = dataset_acts[dataset_idx+traj_idx].unsqueeze(0).numpy()

            # device transfer
            nobs = torch.from_numpy(test_obs).to(device, dtype=torch.float32)

            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                start_time = time.time()

                for k in noise_scheduler.timesteps:
                    # predict noise
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

                end_time = time.time()
                times.append(end_time - start_time)

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]

            expert_actions.append(expert_action)
            nactions.append(naction)

        # visualize trajectory
        print("computation time: ", np.mean(times))
        visualize_trajectory(expert_actions, nactions, nobs)

def visualize_trajectory(expert_actions, action_preds, nobs):

    am = ActionManager() # get action manager
    om = ObservationManager() # get observation manager

    # plot 
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')

    assert len(action_preds) == len(expert_actions), "the number of predicted trajectories and true trajectories should be the same"

    idx = 0
    for action_pred, expert_action in zip(action_preds, expert_actions):

        pred_traj = am.denormalizeTraj(action_pred)
        true_traj = am.denormalizeTraj(expert_action)

        # convert the trajectory to a b-spline
        start_state = getZeroState()
        w_posBS_pred, w_yawBS_pred = am.f_trajAnd_w_State2wBS(pred_traj, start_state)
        w_posBS_true, w_yawBS_true = am.f_trajAnd_w_State2wBS(true_traj, start_state)
        num_vectors_pos = 100
        num_vectors_yaw = 10
        time_pred = np.linspace(w_posBS_pred.getT0(), w_posBS_pred.getTf(), num_vectors_pos)
        time_yaw_pred = np.linspace(w_yawBS_pred.getT0(), w_yawBS_pred.getTf(), num_vectors_yaw)
        time_true = np.linspace(w_posBS_true.getT0(), w_posBS_true.getTf(), num_vectors_pos)
        time_yaw_true = np.linspace(w_yawBS_true.getT0(), w_yawBS_true.getTf(), num_vectors_yaw)

        # plot the predicted trajectory
        if idx == 0:
            ax.plot(w_posBS_true.pos_bs[0](time_true), w_posBS_true.pos_bs[1](time_true), w_posBS_true.pos_bs[2](time_true), lw=4, alpha=0.7, label='Expert', c='green')
            ax.plot(w_posBS_pred.pos_bs[0](time_pred), w_posBS_pred.pos_bs[1](time_pred), w_posBS_pred.pos_bs[2](time_pred), lw=4, alpha=0.7, label='GNN', c='orange')

            # plot the start and goal position
            ax.scatter(w_posBS_true.pos_bs[0](w_posBS_true.getT0()), w_posBS_true.pos_bs[1](w_posBS_true.getT0()), w_posBS_true.pos_bs[2](w_posBS_true.getT0()), s=100, c='pink', marker='o', label='Start')

        else:
            ax.plot(w_posBS_true.pos_bs[0](time_true), w_posBS_true.pos_bs[1](time_true), w_posBS_true.pos_bs[2](time_true), lw=4, alpha=0.7, c='green')
            ax.plot(w_posBS_pred.pos_bs[0](time_pred), w_posBS_pred.pos_bs[1](time_pred), w_posBS_pred.pos_bs[2](time_pred), lw=4, alpha=0.7, c='orange')

        idx += 1
        if idx > len(action_preds):
            break

    # plot the goal
    f_obs = om.denormalizeObservation(nobs.to('cpu').numpy())
    ax.scatter(f_obs[0][7], f_obs[0][8], f_obs[0][9], s=100, c='red', marker='*', label='Goal')
    
    # plot the obstacles
    # get w pos of the obstacles
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

    ax.grid(True)
    ax.legend(loc='best')
    ax.set_xlim(-2, 15)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    # plt.show()

    # fig.savefig('test_pos.png')

    # plot the yaw trajectory
    ax = fig.add_subplot(212)

    idx = 0
    for action_pred, expert_action in zip(action_preds, expert_actions):

        pred_traj = am.denormalizeTraj(action_pred)
        true_traj = am.denormalizeTraj(expert_action)

        # convert the trajectory to a b-spline
        start_state = getZeroState()
        w_posBS_pred, w_yawBS_pred = am.f_trajAnd_w_State2wBS(pred_traj, start_state)
        w_posBS_true, w_yawBS_true = am.f_trajAnd_w_State2wBS(true_traj, start_state)

        if idx == 0:
            ax.plot(time_yaw_true, w_yawBS_true.pos_bs[0](time_yaw_true), lw=4, alpha=0.7, label='Expert', c='green')
            ax.plot(time_yaw_pred, w_yawBS_pred.pos_bs[0](time_yaw_pred), lw=4, alpha=0.7, label='Diffusion', c='orange')
        else:
            ax.plot(time_yaw_true, w_yawBS_true.pos_bs[0](time_yaw_true), lw=4, alpha=0.7, c='green')
            ax.plot(time_yaw_pred, w_yawBS_pred.pos_bs[0](time_yaw_pred), lw=4, alpha=0.7, c='orange')

        idx += 1
        if idx > len(action_preds):
            break

    ax.grid(True)
    ax.legend(loc='best')
    ax.set_xlabel('Time')
    ax.set_ylabel('Yaw')
    # fig.savefig('/home/kota/Research/deep-panther_ws/src/deep_panther/panther/matlab/figures/test_yaw.png')
    plt.show()

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

if __name__ == '__main__':
    main()