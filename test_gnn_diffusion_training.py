#!/usr/bin/env python3

# diffusion policy import
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

# import
import os
import argparse

# network utils import
from utils import ConditionalUnet1D, create_dataset, create_gnn_dataset
from torch_geometric.loader import DataLoader as GNNDataLoader

# wanb
import wandb
wandb.login()

def train(num_epochs, dataloader, device, noise_pred_net, noise_scheduler, ema, optimizer, lr_scheduler, use_gnn):

    """
    Train noise_pred_net
    @param num_epochs: number of epochs
    @param dataloader: dataloader
    @param device: device to transfer data to
    @param noise_pred_net: noise prediction network
    @param noise_scheduler: noise scheduler
    @param ema: Exponential Moving Average
    @param optimizer: optimizer
    @param lr_scheduler: learning rate scheduler
    """

    # training loop
    wandb.init(project='diffusion')
    counter = 0
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:

                    # data reshape
                    nobs = nbatch.obs if use_gnn else nbatch[0].to(device).unsqueeze(1)
                    naction = nbatch.acts if use_gnn else nbatch[1].to(device).unsqueeze(1)
                    x_dict = nbatch.x_dict if use_gnn else None
                    edge_index_dict = nbatch.edge_index_dict if use_gnn else None
                    B = nobs.shape[0]

                    # observation as FiLM conditioning
                    # (B, obs_horizon, obs_dim)
                    obs_cond = nobs[:, :] 
                    # (B, obs_horizon * obs_dim)
                    if not use_gnn:
                        obs_cond = obs_cond.flatten(start_dim=1)

                    # (B, pred_horizon=1, action_dim)
                    if use_gnn:
                        naction = naction.unsqueeze(1)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)
                    
                    # predict the noise residual
                    if use_gnn:
                        # if use gnn, pass x_dict and edge_index_dict
                        noise_pred = noise_pred_net(noisy_actions, timesteps, x_dict=x_dict, edge_index_dict=edge_index_dict)
                    else:
                        # if not use gnn, pass global_cond
                        noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond) 

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(noise_pred_net.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

                    # wandb logging
                    wandb.log({'loss': loss_cpu, 'epoch': epoch_idx})

                # save model
                torch.save(noise_pred_net.state_dict(), f'models/num_{counter}_noise_pred_net.pth')
                counter += 1
                    
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    return noise_pred_net

# def train(num_epochs, dataloader, device, noise_pred_net, noise_scheduler, ema, optimizer, lr_scheduler):

#     """
#     Train noise_pred_net
#     @param num_epochs: number of epochs
#     @param dataloader: dataloader
#     @param device: device to transfer data to
#     @param noise_pred_net: noise prediction network
#     @param noise_scheduler: noise scheduler
#     @param ema: Exponential Moving Average
#     @param optimizer: optimizer
#     @param lr_scheduler: learning rate scheduler
#     """

#     # training loop
#     wandb.init(project='diffusion')
#     counter = 0
#     with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        
#         # epoch loop
#         for epoch_idx in tglobal:
#             epoch_loss = list()
            
#             # batch loop
#             with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
#                 for nbatch in tepoch:

#                     # data reshape
#                     nobs = nbatch[0].to(device).unsqueeze(1)
#                     naction = nbatch[1].to(device).unsqueeze(1)
#                     B = nobs.shape[0]

#                     # observation as FiLM conditioning
#                     # (B, obs_horizon, obs_dim)
#                     obs_cond = nobs[:, :]
#                     # (B, obs_horizon * obs_dim)
#                     obs_cond = obs_cond.flatten(start_dim=1)

#                     # sample noise to add to actions
#                     noise = torch.randn(naction.shape, device=device)

#                     # sample a diffusion iteration for each data point
#                     timesteps = torch.randint(
#                         0, noise_scheduler.config.num_train_timesteps,
#                         (B,), device=device
#                     ).long()

#                     # add noise to the clean images according to the noise magnitude at each diffusion iteration
#                     # (this is the forward diffusion process)
#                     noisy_actions = noise_scheduler.add_noise(
#                         naction, noise, timesteps)

#                     # predict the noise residual
#                     noise_pred = noise_pred_net(
#                         noisy_actions, timesteps, global_cond=obs_cond)

#                     # L2 loss
#                     loss = nn.functional.mse_loss(noise_pred, noise)

#                     # optimize
#                     loss.backward()
#                     optimizer.step()
#                     optimizer.zero_grad()
#                     # step lr scheduler every batch
#                     # this is different from standard pytorch behavior
#                     lr_scheduler.step()

#                     # update Exponential Moving Average of the model weights
#                     ema.step(noise_pred_net.parameters())

#                     # logging
#                     loss_cpu = loss.item()
#                     epoch_loss.append(loss_cpu)
#                     tepoch.set_postfix(loss=loss_cpu)

#                     # wandb logging
#                     wandb.log({'loss': loss_cpu, 'epoch': epoch_idx})

#                 # save model
#                 torch.save(noise_pred_net.state_dict(), f'models/num_{counter}_noise_pred_net.pth')
#                 counter += 1
                    
#             tglobal.set_postfix(loss=np.mean(epoch_loss))

#     return noise_pred_net

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def main():

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='cpu', help='device', type=str)
    parser.add_argument('-g', '--gnn', default=True, help='use gnn', type=str2bool)
    args = parser.parse_args()
    "********************* DATA *********************"

    # device
    device = torch.device(args.device)

    # use gnn?
    use_gnn = args.gnn
    print("Use GNN: ", use_gnn)

    # list npz files in the directory
    # dirs = "/home/jtorde/Research/puma_ws/src/puma/panther_compression/evals/tmp_dagger/2/demos/"
    dirs = "/home/jtorde/Research/puma_ws/src/puma/panther_compression/evals-dir/evals4/tmp_dagger/2/demos/"

    # create dataset
    dataset = create_gnn_dataset(dirs, device) if use_gnn else create_dataset(dirs, device)

    # parameters (from diffuion policy paper)
    pred_horizon = 1 
    obs_horizon = 1

    # create dataloader
    if not use_gnn:
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
    else:
        # create dataloader for GNN
        dataloader = GNNDataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=1,
            persistent_workers=True
        )

    "********************* NETWORK *********************"

    # parameters
    # for now we don't use GNN
    obs_dim = 43
    action_dim = 22 # 15(pos) + 6(yaw) + 1(time)

    # hyperparameters
    num_diffusion_iters = 100
    
    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon,
        gnn_data=dataset[0] if use_gnn else None,
        use_gnn=use_gnn
    )

    # example inputs
    noised_action = torch.randn((1, pred_horizon, action_dim))
    obs = torch.zeros((1, obs_horizon, obs_dim))

    # example diffusion iteration
    diffusion_iter = torch.zeros((1,))

    # the noise prediction network
    # takes noisy action, diffusion iteration and observation as input
    # predicts the noise added to action
    # you need to run noise_pred_net to initialize the network https://stackoverflow.com/questions/75550160/how-to-set-requires-grad-to-false-freeze-pytorch-lazy-layers
    x_dict = dataset[0].x_dict if use_gnn else None
    edge_index_dict = dataset[0].edge_index_dict if use_gnn else None
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
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    _ = noise_pred_net.to(device)

    "********************* TRAINING LOOP *********************"

    # hyperparameters
    num_epochs = 500

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=noise_pred_net.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    # training loop
    noise_pred_net = train(
        num_epochs=num_epochs,
        dataloader=dataloader,
        device=device,
        noise_pred_net=noise_pred_net,
        noise_scheduler=noise_scheduler,
        ema=ema,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        use_gnn=use_gnn
    )

    # Weights of the EMA model
    # is used for inference
    ema_noise_pred_net = noise_pred_net
    ema.copy_to(ema_noise_pred_net.parameters())

    # save model
    torch.save(ema_noise_pred_net.state_dict(), 'models/final_ema_noise_pred_net.pth')

if __name__ == '__main__':
    main()