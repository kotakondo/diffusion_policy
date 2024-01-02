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
            **kwargs):
        
        super().__init__()

        # unpack kwargs
        kernel_size = kwargs["diffusion_kernel_size"]
        n_groups = kwargs["diffusion_n_groups"]
        use_gnn = kwargs["en_network_type"] == "gnn"
        lstm_hidden_size = kwargs["lstm_hidden_size"]
        transformer_dim_feedforward = kwargs["transformer_dim_feedforward"]
        transformer_dropout = kwargs["transformer_dropout"]
        gnn_hidden_channels = kwargs["gnn_hidden_channels"]
        gnn_num_layers = kwargs["gnn_num_layers"]
        gnn_num_heads = kwargs["gnn_num_heads"]
        group = kwargs["gnn_group"]
        gnn_data = kwargs["gnn_data"]
        en_network_type = kwargs["en_network_type"]
        mlp_hidden_sizes = kwargs["mlp_hidden_sizes"]
        mlp_activation = kwargs["mlp_activation"]

        self.use_gnn = use_gnn
        self.gnn_data = gnn_data
        self.gnn_hidden_channels = gnn_hidden_channels
        self.gnn_num_layers = gnn_num_layers
        self.gnn_num_heads = gnn_num_heads
        self.group = group
        self.out_channels = out_channels
        self.en_network_type = en_network_type
        self.mlp_hidden_sizes = mlp_hidden_sizes
        self.mlp_activation = mlp_activation

        # Use LSTM for encoder
        if self.en_network_type == "lstm":
            self.lstm = nn.LSTM(cond_dim, lstm_hidden_size, batch_first=True)
            linear_layer_input_dim = lstm_hidden_size

        # Use Transformer for encoder
        if self.en_network_type == "transformer":
            transformer_nhead = 13 # 1, 13, 23, or 299
            self.transformer = nn.TransformerEncoderLayer(d_model=cond_dim, nhead=transformer_nhead, dim_feedforward=transformer_dim_feedforward, dropout=transformer_dropout, batch_first=True)
            linear_layer_input_dim = cond_dim

        # Use GNN for encoder
        if self.en_network_type == "gnn":
            self.lin_dict = th.nn.ModuleDict()
            for node_type in self.gnn_data.node_types:
                self.lin_dict[node_type] = gnn_Linear(-1, self.gnn_hidden_channels)
            # HGTConv Layers
            self.convs = th.nn.ModuleList()
            for _ in range(self.gnn_num_layers):
                conv = HGTConv(self.gnn_hidden_channels, self.gnn_hidden_channels, self.gnn_data.metadata(), self.gnn_num_heads, group=self.group)
                self.convs.append(conv)
            linear_layer_input_dim = self.gnn_hidden_channels

        # linear layers
        layers = []
        for next_dim in mlp_hidden_sizes:
            layers.append(nn.Linear(linear_layer_input_dim, next_dim))
            layers.append(mlp_activation)
            linear_layer_input_dim = next_dim
        layers.append(nn.Linear(linear_layer_input_dim, cond_dim))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)
        self.tanh = nn.Tanh()

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

        # Use LSTM for encoder
        if self.en_network_type == "lstm":
            output, (h_n, c_n) = self.lstm(cond) # get the output
            if len(output.shape) == 3:
                output = output[:, -1, :]
            else:
                output = output[[-1], :]
            # output = output[:, -1, :] # get the last output
            output = self.layers(output) # pass it through the layers
            cond = self.tanh(output) # pass it through a tanh layer
        
        elif self.en_network_type == "transformer":
            output = self.transformer(cond) # get the output
            output = self.layers(output) # pass it through the layers
            cond = self.tanh(output) # pass it through a tanh layer

        # Use GNN for encoder
        if self.en_network_type == "gnn" and x_dict is not None and edge_index_dict is not None:
            for node_type, x_gnn in x_dict.items():
                x_dict[node_type] = self.lin_dict[node_type](x_gnn).relu_()
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
            output = x_dict["current_state"] # extract the latent vector
            output = self.layers(output)
            cond = self.tanh(output) # pass it through a tanh layer

        # FiLM modulation
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
    def __init__(self, **kwargs):
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

        # unpack kwargs
        input_dim = kwargs["action_dim"]
        diffusion_step_embed_dim = kwargs["diffusion_step_embed_dim"]
        down_dims = kwargs["diffusion_down_dims"]
        kernel_size = kwargs["diffusion_kernel_size"]
        global_cond_dim = kwargs["obs_dim"] * kwargs["obs_horizon"]

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

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out*2, dim_in, cond_dim=cond_dim, **kwargs),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, **kwargs),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, **kwargs),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, **kwargs),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, **kwargs),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, **kwargs),
                Downsample1d(dim_out) if not is_last else nn.Identity()
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
    def __init__(self, **kwargs):
        
        super().__init__()

        # unpack kwargs
        num_trajs = kwargs["num_trajs"]
        input_dim = kwargs["input_dim"]
        output_dim = kwargs["output_dim"]
        self.en_network_type = kwargs["en_network_type"]
        mlp_hidden_sizes = kwargs["mlp_hidden_sizes"]
        mlp_activation = kwargs["mlp_activation"]
        lstm_hidden_size = kwargs["lstm_hidden_size"]
        transformer_d_model = kwargs["transformer_d_model"]
        transformer_nhead = kwargs["transformer_nhead"]
        transformer_dim_feedforward = kwargs["transformer_dim_feedforward"]
        transformer_dropout = kwargs["transformer_dropout"]
        gnn_data = kwargs["gnn_data"]
        gnn_hidden_channels = kwargs["gnn_hidden_channels"]
        gnn_num_layers = kwargs["gnn_num_layers"]
        gnn_num_heads = kwargs["gnn_num_heads"]
        gnn_group = kwargs["gnn_group"]

        self.num_trajs = num_trajs
        self.output_dim = output_dim
        
        # create a list of layers

        if self.en_network_type == "lstm":
            self.lstm = nn.LSTM(input_dim, lstm_hidden_size, batch_first=True)
            input_dim = lstm_hidden_size
        
        elif self.en_network_type == "transformer":
            self.transformer = nn.TransformerEncoderLayer(d_model=transformer_d_model, nhead=transformer_nhead, dim_feedforward=transformer_dim_feedforward, dropout=transformer_dropout, batch_first=True)
            input_dim = transformer_d_model

        elif self.en_network_type == "gnn":

            self.lin_dict = th.nn.ModuleDict()
            for node_type in gnn_data.node_types:
                self.lin_dict[node_type] = gnn_Linear(-1, gnn_hidden_channels)

            # HGTConv Layers
            self.convs = th.nn.ModuleList()
            for _ in range(gnn_num_layers):
                conv = HGTConv(gnn_hidden_channels, gnn_hidden_channels, gnn_data.metadata(), gnn_num_heads, group=gnn_group)
                self.convs.append(conv)
            input_dim = gnn_hidden_channels

        layers = []
        for next_dim in mlp_hidden_sizes:
            layers.append(nn.Linear(input_dim, next_dim))
            layers.append(mlp_activation)
            input_dim = next_dim 
        layers.append(nn.Linear(input_dim, self.output_dim*self.num_trajs))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)
        self.tanh = nn.Tanh()

    def forward(self, x: th.Tensor, x_dict=None, edge_index_dict=None) -> th.Tensor:

        if self.en_network_type == "mlp":

            batch_size = x.shape[0]
            output = self.layers(x)
            output = self.tanh(output) # pass it through a tanh layer
            output = output.reshape(batch_size, self.num_trajs, self.output_dim) # reshape output to (batch_size, num_trajs, output_dim)
        
        elif self.en_network_type == "lstm":

            output, (h_n, c_n) = self.lstm(x) # get the output
            output = output[:, -1, :] # get the last output
            output = self.layers(output) # pass it through the layers
            output = self.tanh(output) # pass it through a tanh layer
            output = output.reshape(output.shape[0], self.num_trajs, self.output_dim) # reshape output to (batch_size, num_trajs, output_dim)
        
        elif self.en_network_type == "transformer":

            output = self.transformer(x) # get the output
            output = self.layers(output) # pass it through the layers
            output = self.tanh(output) # pass it through a tanh layer
            output = output.reshape(output.shape[0], self.num_trajs, self.output_dim) # reshape output to (batch_size, num_trajs, output_dim)

        elif self.en_network_type == "gnn":

            for node_type, x_gnn in x_dict.items():
                x_dict[node_type] = self.lin_dict[node_type](x_gnn).relu_()

            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)

            output = x_dict["current_state"] # extract the latent vector
            output = self.layers(output) # pass it through the layers
            output = self.tanh(output)
            output = output.reshape(output.shape[0], self.num_trajs, self.output_dim)

        return output
