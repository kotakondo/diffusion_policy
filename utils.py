#/usr/bin/env python3

# diffusion policy import
from typing import Union
import math
import torch
import torch.nn as nn
import os
import time
import numpy as np
import argparse
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# torch import
import torch as th
from torch.utils.data import TensorDataset

# gnn import
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.nn import Linear as gnn_Linear

# visualization import
import matplotlib.pyplot as plt
from compression.utils.other import ObservationManager, ActionManager, getZeroState
from tqdm import tqdm

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
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
            n_groups=8,
            use_gnn=True,
            gnn_data=None,
            gnn_hidden_channels=128,
            gnn_num_layers=4,
            gnn_num_heads=8,
            group='max',
            num_linear_layers=2,
            linear_hidden_channels=2048,
            ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        cond_dim = gnn_hidden_channels if use_gnn else cond_dim
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        self.use_gnn = use_gnn
        self.gnn_data = gnn_data
        self.gnn_hidden_channels = gnn_hidden_channels
        self.gnn_num_layers = gnn_num_layers
        self.gnn_num_heads = gnn_num_heads
        self.group = group
        self.num_linear_layers = num_linear_layers
        self.linear_hidden_channels = linear_hidden_channels
        self.out_channels = out_channels

        if self.use_gnn:
            self.lin_dict = th.nn.ModuleDict()
            for node_type in self.gnn_data.node_types:
                self.lin_dict[node_type] = gnn_Linear(-1, self.gnn_hidden_channels)

            # HGTConv Layers
            self.convs = th.nn.ModuleList()
            for _ in range(self.gnn_num_layers):
                conv = HGTConv(self.gnn_hidden_channels, self.gnn_hidden_channels, self.gnn_data.metadata(), self.gnn_num_heads, group=self.group)
                self.convs.append(conv)

            # add linear layers (num_linear_layers) times
            # self.lins = th.nn.ModuleList()
            # for _ in range(num_linear_layers-1):
            #     self.lins.append(gnn_Linear(-1, linear_hidden_channels)) 
            # self.lins.append(gnn_Linear(-1, self.out_channels))

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        
        # make it float
        self = self.float()

    def forward(self, x, cond=None, x_dict=None, edge_index_dict=None):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)

        if self.use_gnn and x_dict is not None and edge_index_dict is not None:

            for node_type, x_gnn in x_dict.items():
                # if x_gnn is double then convert it to float
                # if type(x_gnn) is th.Tensor:
                #     if x_gnn.dtype == th.double:
                #         x_gnn = x_gnn.float()

                # print out what device x_gnn is on
                x_dict[node_type] = self.lin_dict[node_type](x_gnn).relu_()

            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)

            # extract the latent vector
            cond = x_dict["current_state"]

            # add linear layers
            # for lin in self.lins:
            #     cond = lin(cond)
        
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
        gnn_data,
        use_gnn,
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
                kernel_size=kernel_size, n_groups=n_groups, gnn_data=gnn_data, use_gnn=use_gnn),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups, gnn_data=gnn_data, use_gnn=use_gnn),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups, gnn_data=gnn_data, use_gnn=use_gnn),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups, gnn_data=gnn_data, use_gnn=use_gnn),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups, gnn_data=gnn_data, use_gnn=use_gnn),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups, gnn_data=gnn_data, use_gnn=use_gnn),
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

        # print("number of parameters: {:e}".format(
        #     sum(p.numel() for p in self.parameters()))
        # )

        # make it float
        self = self.float()

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None,
            x_dict=None,
            edge_index_dict=None
            ):
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

            # print out devices
            x = resnet(x, global_feature, x_dict, edge_index_dict)
            x = resnet2(x, global_feature, x_dict, edge_index_dict)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature, x_dict, edge_index_dict)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature, x_dict, edge_index_dict)
            x = resnet2(x, global_feature, x_dict, edge_index_dict)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x

def network_init(action_dim, obs_dim, obs_horizon, pred_horizon, num_diffusion_iters, dataset_training, use_gnn, device):

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon,
        gnn_data=dataset_training[0] if use_gnn else None,
        use_gnn=use_gnn,
    )

    # example inputs
    noised_action = torch.randn((1, pred_horizon, action_dim)).to(device)
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

    # for this demo, we use DDPMScheduler with 100 diffusion iterations
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

    # loop over dirs
    obs_data = th.tensor([]).to(device)
    traj_data = th.tensor([]).to(device)

    # list dirs in dirs
    dirs = [ f.path for f in os.scandir(dirs) if f.is_dir() ]
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

    # print out the total data size
    print(f"original data size: {obs_data.shape[0] * traj_data.shape[1]}") # obs_data.shape[0] is the num of demos, traj_data.shape[1] is num of trajs per each demo (default is 10)

    # rearanage the data
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

def get_nactions(noise_pred_net, noise_scheduler, dataset, pred_horizon, num_diffusion_iters, action_dim, use_gnn, device, is_visualize=True, num_eval=None):

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
    for dataset_idx in tqdm(range(num_data_to_load), desc="data idx"):

        # stack the last obs_horizon (2) number of observations
        nobs = dataset[dataset_idx].obs if use_gnn else dataset[dataset_idx][0].unsqueeze(0)
        expert_action = dataset[dataset_idx].acts if use_gnn else dataset[dataset_idx][1].unsqueeze(0)

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

            # start timer
            start_time = time.time()

            for k in tqdm(noise_scheduler.timesteps, desc="diffusion iter k"):
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

        # print out the computation time
        print("computation time: ", np.mean(times))

        # visualize trajectory
        if is_visualize:
            visualize_trajectory(expert_actions, nactions, nobs, dataset_idx)

    if not is_visualize: # used get_nactions used in evaluation
        return expert_actions, nactions, nobses

def visualize_trajectory(expert_actions, action_preds, nobs, image_idx):

    # get action and observation manager for normalization and denormalization
    am = ActionManager() # get action manager
    om = ObservationManager() # get observation manager

    # plot 
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')

    assert len(action_preds) == len(expert_actions), "the number of predicted trajectories and true trajectories should be the same"

    # position trajectory
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
    fig.savefig(f'/media/jtorde/T7/gdp/pngs/image_{image_idx}.png')
    # plt.show()