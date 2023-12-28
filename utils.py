#/usr/bin/env python3

# diffusion policy import
from typing import Union
import math
import torch
import torch.nn as nn
import os
import numpy as np
import argparse

# torch import
import torch as th
from torch.utils.data import TensorDataset

# gnn import
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.nn import Linear as gnn_Linear

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
        n_groups=8,
        device='cuda'
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
            nn.Linear(dsed, dsed * 4, device=device),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed, device=device),
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

def create_pair_obs_act(dirs, device, is_visualize=False):

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

    print("dataset before rearranging: ")
    print("obs_data.shape: ", obs_data.shape)
    print("traj_data.shape: ", traj_data.shape)

    assert obs_data.shape[0] == traj_data.shape[0], "obs_data and traj_data must have the same number of samples"

    dataset_obs = th.tensor([]).to(device)
    dataset_acts = th.tensor([]).to(device)
    idx = 0
    for i in range(obs_data.shape[0]): # loop over samples
        for j in range(traj_data.shape[1]): # loop over expert demonstrations (10)
            dataset_obs = th.cat((dataset_obs, obs_data[i, 0, :].unsqueeze(0).unsqueeze(0)), 0)
            dataset_acts = th.cat((dataset_acts, traj_data[i, j, :].unsqueeze(0).unsqueeze(0)), 0)
            idx += 1
            if idx >= 1000 and is_visualize:
                break
        if idx >= 1000 and is_visualize:
            break

    dataset_obs = dataset_obs.squeeze(1)
    dataset_acts = dataset_acts.squeeze(1)

    print("dataset after rearranging: ")
    print("dataset_obs.shape: ", dataset_obs.shape)
    print("dataset_acts.shape: ", dataset_acts.shape)

    return dataset_obs, dataset_acts


def create_dataset(dirs, device, is_visualize=False):
    
    """
    Create dataset from npz files in dirs
    @param dirs: directory containing npz files
    @param device: device to transfer data to
    @return dataset: TensorDataset
    """
    
    # get obs and acts
    dataset_obs, dataset_acts = create_pair_obs_act(dirs, device, is_visualize)

    # create dataset
    dataset = TensorDataset(dataset_obs, dataset_acts)

    return dataset

def create_gnn_dataset(dirs, device, is_visualize=False):

    """
    This function generates a dataset for GNN
    """

    " ********************* GET DATA ********************* "

    # get obs and acts
    dataset_obs, dataset_acts = create_pair_obs_act(dirs, device, is_visualize)

    " ********************* INITIALIZE DATASET ********************* "

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
                    dist_obst_to_obst.append((np.linalg.norm(feature_vector_for_obs[33*j:33*j+3] - feature_vector_for_obs[33*k:33*k+3])).to('cpu').numpy())

        " ********************* MAKE A DATA OBJECT FOR HETEROGENEUS GRAPH ********************* "

        # add nodes
        data["current_state"].x = feature_vector_for_current_state.unsqueeze(0).float().to(device)
        data["goal_state"].x = feature_vector_for_goal.unsqueeze(0).float().to(device)
        data["observation"].x = th.stack([feature_vector_for_obs[33*j:33*(j+1)] for j in range(num_of_obst)], dim=0).float().to(device)

        # add edges
        # data["current_state", "dist_current_state_to_goal_state", "goal_state"].edge_index = th.tensor([
        #                                                                             [0],  # idx of source nodes (current state)
        #                                                                             [0],  # idx of target nodes (goal state)
        #                                                                             ],dtype=th.int64)
        # data["current_state", "dist_current_state_to_observation", "observation"].edge_index = th.tensor([
        #                                                                                 [0],  # idx of source nodes (current state)
        #                                                                                 [0],  # idx of target nodes (observation)
        #                                                                                 ],dtype=th.int64)
        # data["observation", "dist_obs_to_goal", "goal_state"].edge_index = th.tensor([
        #                                                                                 [0, 0],  # idx of source nodes (observation)
        #                                                                                 [0, 1],  # idx of target nodes (goal state)
        #                                                                                 ],dtype=th.int64)
        # data["observation", "dist_observation_to_current_state", "current_state"].edge_index = th.tensor([
        #                                                                                 [0, 0],  # idx of source nodes (observation)
        #                                                                                 [0, 1],  # idx of target nodes (current state)
        #                                                                                 ],dtype=th.int64)
        # data["goal_state", "dist_goal_state_to_current_state", "current_state"].edge_index = th.tensor([
        #                                                                                 [0],  # idx of source nodes (goal state)
        #                                                                                 [0],  # idx of target nodes (current state)
        #                                                                                 ],dtype=th.int64)
        # data["goal_state", "dist_to_obs", "observation"].edge_index = th.tensor([
        #                                                                                 [0, 0],  # idx of source nodes (goal state)
        #                                                                                 [0, 1],  # idx of target nodes (observation)
        #                                                                                 ],dtype=th.int64)

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

# def create_gnn_dataset(dirs, device, is_visualize=False):

#     """
#     This function generates a dataset for GNN
#     """

#     " ********************* GET DATA ********************* "

#     # get obs and acts
#     dataset_obs, dataset_acts = create_pair_obs_act(dirs, device, is_visualize)

#     " ********************* INITIALIZE DATASET ********************* "

#     dataset = []

#     assert dataset_obs.shape[0] == dataset_acts.shape[0], "the length of dataset_obs and dataset_acts should be the same"

#     for i in range(dataset_obs.shape[0]):

#         " ********************* GET NODES ********************* "

#         # nodes you need for GNN
#         # dataset = [f_v, f_a, yaw_dot, f_g,  [f_ctrl_pts_o0], bbox_o0, [f_ctrl_pts_o1], bbox_o1 ,...]
#         # dim         3    3      1      3          30            3          30              3 
#         # 0. current state 
#         # 1. goal state 
#         # 2. observation
#         # In the current setting f_obs is a realative state from the current state so we pass f_v, f_z, yaw_dot to the current state node

#         feature_vector_for_current_state = dataset_obs[i][0:7]
#         feature_vector_for_goal = dataset_obs[i][7:10]
#         feature_vector_for_obs = dataset_obs[i][10:]

#         " ********************* MAKE A DATA OBJECT FOR HETEROGENEUS GRAPH ********************* "

#         data = HeteroData()

#         # add nodes
#         # get num of obst
#         num_of_obst = int(len(dataset_obs[i][10:])/33)

#         # if type(dataset_obs[i]) is np.ndarray:
#         #     warnings.warn("f_obs_n is a numpy array - converting it to a torch tensor")
#         #     dataset_obs[i] = th.tensor(dataset_obs[i], dtype=th.double).to(device)

#         dist_current_state_goal = th.tensor(np.linalg.norm(feature_vector_for_goal[:3].to('cpu').numpy())).to(device)

#         dist_current_state_obs = []
#         dist_goal_obs = []
#         for j in range(num_of_obst):
#             dist_current_state_obs.append(np.linalg.norm(feature_vector_for_obs[33*j:33*j+3].to('cpu').numpy()))
#             dist_goal_obs.append(np.linalg.norm((feature_vector_for_goal[:3] - feature_vector_for_obs[33*j:33*j+3]).to('cpu').numpy()))

#         dist_current_state_obs = th.tensor(dist_current_state_obs, dtype=th.float).to(device)
#         dist_goal_obs = th.tensor(dist_goal_obs, dtype=th.float).to(device)

#         dist_obst_to_obst = []
#         for j in range(num_of_obst):
#             for k in range(num_of_obst):
#                 if j != k:
#                     dist_obst_to_obst.append(np.linalg.norm(feature_vector_for_obs[33*j:33*j+3] - feature_vector_for_obs[33*k:33*k+3]))
        
#         dist_obst_to_obst = th.tensor(dist_obst_to_obst, dtype=th.float).to(device)

#         " ********************* MAKE A DATA OBJECT FOR HETEROGENEUS GRAPH ********************* "

#         # add nodes
#         data["current_state"].x = feature_vector_for_current_state.float().unsqueeze(0).to(device)
#         data["goal_state"].x = feature_vector_for_goal.float().unsqueeze(0).to(device)
#         data["observation"].x = th.stack([feature_vector_for_obs[33*j:33*(j+1)] for j in range(num_of_obst)], dim=0).float().unsqueeze(0).to(device)

#         # add edges
#         # data["current_state", "dist_current_state_to_goal_state", "goal_state"].edge_index = th.tensor([
#         #                                                                             [0],  # idx of source nodes (current state)
#         #                                                                             [0],  # idx of target nodes (goal state)
#         #                                                                             ],dtype=th.int64)
#         # data["current_state", "dist_current_state_to_observation", "observation"].edge_index = th.tensor([
#         #                                                                                 [0],  # idx of source nodes (current state)
#         #                                                                                 [0],  # idx of target nodes (observation)
#         #                                                                                 ],dtype=th.int64)
#         # data["observation", "dist_obs_to_goal", "goal_state"].edge_index = th.tensor([
#         #                                                                                 [0, 0],  # idx of source nodes (observation)
#         #                                                                                 [0, 1],  # idx of target nodes (goal state)
#         #                                                                                 ],dtype=th.int64)
#         # data["observation", "dist_observation_to_current_state", "current_state"].edge_index = th.tensor([
#         #                                                                                 [0, 0],  # idx of source nodes (observation)
#         #                                                                                 [0, 1],  # idx of target nodes (current state)
#         #                                                                                 ],dtype=th.int64)
#         # data["goal_state", "dist_goal_state_to_current_state", "current_state"].edge_index = th.tensor([
#         #                                                                                 [0],  # idx of source nodes (goal state)
#         #                                                                                 [0],  # idx of target nodes (current state)
#         #                                                                                 ],dtype=th.int64)
#         # data["goal_state", "dist_to_obs", "observation"].edge_index = th.tensor([
#         #                                                                                 [0, 0],  # idx of source nodes (goal state)
#         #                                                                                 [0, 1],  # idx of target nodes (observation)
#         #                                                                                 ],dtype=th.int64)

#         data["current_state", "dist_current_state_to_goal_state", "goal_state"].edge_index = th.tensor([
#                                                                                     [0],  # idx of source nodes (current state)
#                                                                                     [0],  # idx of target nodes (goal state)
#                                                                                     ],dtype=th.int64)
#         data["current_state", "dist_current_state_to_observation", "observation"].edge_index = th.tensor([
#                                                                                         [0],  # idx of source nodes (current state)
#                                                                                         [0],  # idx of target nodes (observation)
#                                                                                         ],dtype=th.int64)
#         data["observation", "dist_obs_to_goal", "goal_state"].edge_index = th.tensor([
#                                                                                         [0],  # idx of source nodes (observation)
#                                                                                         [0],  # idx of target nodes (goal state)
#                                                                                         ],dtype=th.int64)
#         data["observation", "dist_observation_to_current_state", "current_state"].edge_index = th.tensor([
#                                                                                         [0],  # idx of source nodes (observation)
#                                                                                         [0],  # idx of target nodes (current state)
#                                                                                         ],dtype=th.int64)
#         data["goal_state", "dist_goal_state_to_current_state", "current_state"].edge_index = th.tensor([
#                                                                                         [0],  # idx of source nodes (goal state)
#                                                                                         [0],  # idx of target nodes (current state)
#                                                                                         ],dtype=th.int64)
#         data["goal_state", "dist_to_obs", "observation"].edge_index = th.tensor([
#                                                                                         [0],  # idx of source nodes (goal state)
#                                                                                         [0],  # idx of target nodes (observation)
#                                                                                         ],dtype=th.int64)

#         # add edge weights
#         data["current_state", "dist_current_state_to_goal_state", "goal_state"].edge_attr = dist_current_state_goal
#         data["current_state", "dist_current_state_to_observation", "observation"].edge_attr = dist_current_state_obs
#         data["observation", "dist_obs_to_goal", "goal_state"].edge_attr = dist_goal_obs
#         # make it undirected
#         data["observation", "dist_observation_to_current_state", "current_state"].edge_attr = dist_current_state_obs
#         data["goal_state", "dist_goal_state_to_current_state", "current_state"].edge_attr = dist_current_state_goal
#         data["goal_state", "dist_goal_to_obs", "observation"].edge_attr = dist_goal_obs

#         # add ground truth trajectory
#         data.acts = dataset_acts[i].unsqueeze(0).float().to(device)
        
#         # add observation
#         data.obs = dataset_obs[i].unsqueeze(0).float().to(device)

#         # convert the data to the device
#         data = data.to(device)
#         # append data to the dataset
#         dataset.append(data)

#     " ********************* RETURN ********************* "

#     return dataset