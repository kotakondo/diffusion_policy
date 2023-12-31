#!/usr/bin/env python3

# diffusion policy import
import torch
import torch.nn as nn
import math
from typing import Union

# torch import
import torch as th

# gnn import
from torch_geometric.nn import HGTConv
from torch_geometric.nn import Linear as gnn_Linear


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
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1) # not sure why but the default kernel_size was set to 4
        # self.conv = nn.ConvTranspose1d(dim, dim, 3, 2, 1) # not sure why but the default kernel_size was set to 4

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

class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_trajs: int,
        hidden_sizes: list = [256, 256],
        activation: nn.Module = nn.ReLU()
    ):
        
        super().__init__()

        self.num_trajs = num_trajs
        self.output_dim = output_dim
        self.action_dim = output_dim*num_trajs
        
        # create a list of layers
        layers = []
        in_dim = input_dim
        for i, next_dim in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_dim, next_dim))
            layers.append(activation)
            in_dim = next_dim 
        layers.append(nn.Linear(in_dim, self.action_dim))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        batch_size = x.shape[0]
        output = self.layers(x)
        # reshape output to (batch_size, num_trajs, output_dim)
        output = output.reshape(batch_size, self.num_trajs, self.output_dim)
        return output
    
class LSTM(nn.Module):
    """
    LSTM
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_trajs: int,
        hidden_size: int = 256
    ):
        
        super().__init__()
        
        self.num_trajs = num_trajs
        self.output_dim = output_dim
        self.action_dim = output_dim*num_trajs

        # create a list of layers
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.action_dim)
        self.tanh = nn.Tanh()

    def forward(self, x: th.Tensor) -> th.Tensor:

        # get the output
        output, (h_n, c_n) = self.lstm(x)
        # get the last output
        output = output[:, -1, :]
        # pass it through a linear layer
        output = self.linear(output)
        # pass it through a tanh layer
        output = self.tanh(output)
        # reshape output to (batch_size, num_trajs, output_dim)
        output = output.reshape(output.shape[0], self.num_trajs, self.output_dim)

        return output

class Transformer(nn.Module):
    """
    Transformer
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_trajs: int
    ):
        
        super().__init__()

        self.num_trajs = num_trajs
        self.output_dim = output_dim
        self.action_dim = output_dim*num_trajs
        
        # create a list of layers
        self.transformer = nn.Transformer(d_model=43, nhead=43, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.linear = nn.Linear(input_dim, self.action_dim)
        self.tanh = nn.Tanh()

    def forward(self, x: th.Tensor) -> th.Tensor:

        # get the output
        output = self.transformer(x, x)
        # pass it through a linear layer
        output = self.linear(output)
        # pass it through a tanh layer
        output = self.tanh(output)
        # reshape output to (batch_size, num_trajs, output_dim)
        output = output.reshape(output.shape[0], self.num_trajs, self.output_dim)

        return output