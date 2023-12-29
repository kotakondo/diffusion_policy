#!/usr/bin/env python3

# diffusion policy import
import numpy as np
import torch
import th.nn as nn
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

# import
import torch as th
import os
import argparse
import time

# network utils import
from utils import create_dataset, create_gnn_dataset, str2bool, network_init, get_nactions
from torch_geometric.loader import DataLoader as GNNDataLoader

# import from py_panther
from compression.utils.other import CostComputer

# wanb
import wandb
wandb.login()

# avoid opening too many files
th.multiprocessing.set_sharing_strategy('file_system')

def evaluate(dataset, device, noise_pred_net, noise_scheduler, pred_horizon, num_diffusion_iters, action_dim, use_gnn, num_eval):

    """
    Evaluate noise_pred_net
    @param dataset_eval: evaluation dataset
    @param device: device to transfer data to
    @param noise_pred_net: noise prediction network
    """

    # get cost computer
    cost_computer = CostComputer()

    # get expert actions and student actions
    expert_actions, student_actions, nobs = get_nactions(noise_pred_net, noise_scheduler, dataset, pred_horizon, num_diffusion_iters, action_dim, use_gnn, device, is_visualize=False, num_eval=num_eval)

    # evaluation for expert actions
    cost_expert, obst_avoidance_violation_expert, dyn_lim_violation_expert, augmented_cost_expert = [], [], [], []
    for nob, expert_action in zip(nobs, expert_actions):
        # compute cost
        cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = cost_computer.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(nob, expert_action)
        cost_expert.append(cost)
        obst_avoidance_violation_expert.append(obst_avoidance_violation)
        dyn_lim_violation_expert.append(dyn_lim_violation)
        augmented_cost_expert.append(augmented_cost)
    cost_expert = np.array(cost_expert)
    obst_avoidance_violation_expert = np.array(obst_avoidance_violation_expert)
    dyn_lim_violation_expert = np.array(dyn_lim_violation_expert)
    augmented_cost_expert = np.array(augmented_cost_expert)

    # evaluation for student actions
    cost_student, obst_avoidance_violation_student, dyn_lim_violation_student, augmented_cost_student = [], [], [], []
    for nob, student_action in zip(nobs, student_actions):
        # compute cost
        cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = cost_computer.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(nob, student_action)
        cost_student.append(cost)
        obst_avoidance_violation_student.append(obst_avoidance_violation)
        dyn_lim_violation_student.append(dyn_lim_violation)
        augmented_cost_student.append(augmented_cost)
    cost_student = np.array(cost_student)
    obst_avoidance_violation_student = np.array(obst_avoidance_violation_student)
    dyn_lim_violation_student = np.array(dyn_lim_violation_student)
    augmented_cost_student = np.array(augmented_cost_student)

    # return
    return cost_expert, obst_avoidance_violation_expert, dyn_lim_violation_expert, augmented_cost_expert, cost_student, obst_avoidance_violation_student, dyn_lim_violation_student, augmented_cost_student

def train(num_epochs, dataloader_training, dataset_training, device, noise_pred_net, noise_scheduler, ema, optimizer, lr_scheduler, use_gnn, dataset_eval, pred_horizon, num_diffusion_iters, action_dim, save_dir, num_eval):

    """
    Train noise_pred_net
    @param num_epochs: number of epochs
    @param dataloader_training: dataloader_training
    @param device: device to transfer data to
    @param noise_pred_net: noise prediction network
    @param noise_scheduler: noise scheduler
    @param ema: Exponential Moving Average
    @param optimizer: optimizer
    @param lr_scheduler: learning rate scheduler
    """

    # training loop
    wandb.init(project='diffusion')
    epoch_counter = 0
    overfitting_counter = 0
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            
            # batch loop
            with tqdm(dataloader_training, desc='Batch', leave=False) as tepoch:
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
                    noise = th.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = th.randint(
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
                    loss = loss.item()
                    epoch_loss.append(loss)
                    tepoch.set_postfix(loss=loss)

                    # wandb logging
                    wandb.log({'loss': loss, 'epoch': epoch_idx})

                # save model
                th.save(noise_pred_net.state_dict(), f'{save_dir}/num_{epoch_counter}_noise_pred_net.pth')
                epoch_counter += 1
                    
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # each epoch, we evaluate the model on evaluation data
            costs = evaluate(dataset_eval, device, noise_pred_net, noise_scheduler, pred_horizon, num_diffusion_iters, action_dim, use_gnn, num_eval)
            # unpack
            cost_expert_eval, obst_avoidance_violation_expert_eval, dyn_lim_violation_expert_eval, augmented_cost_expert_eval, cost_student_eval, obst_avoidance_violation_student_eval, dyn_lim_violation_student_eval, augmented_cost_student_eval = costs

            # each epoch we evaluate the model on training data too (to check overfitting)
            costs = evaluate(dataset_training, device, noise_pred_net, noise_scheduler, pred_horizon, num_diffusion_iters, action_dim, use_gnn, num_eval)
            # unpack
            cost_expert_training, obst_avoidance_violation_expert_training, dyn_lim_violation_expert_training, augmented_cost_expert_training, cost_student_training, obst_avoidance_violation_student_training, dyn_lim_violation_student_training, augmented_cost_student_training = costs

            # wandb logging
            wandb.log({
                'cost_expert_eval': cost_expert_eval,
                'obst_avoidance_violation_expert_eval': obst_avoidance_violation_expert_eval,
                'dyn_lim_violation_expert_eval': dyn_lim_violation_expert_eval,
                'augmented_cost_expert_eval': augmented_cost_expert_eval,
                'cost_student_eval': cost_student_eval,
                'obst_avoidance_violation_student_eval': obst_avoidance_violation_student_eval,
                'dyn_lim_violation_student_eval': dyn_lim_violation_student_eval,
                'augmented_cost_student_eval': augmented_cost_student_eval,
                'cost_expert_training': cost_expert_training,
                'obst_avoidance_violation_expert_training': obst_avoidance_violation_expert_training,
                'dyn_lim_violation_expert_training': dyn_lim_violation_expert_training,
                'augmented_cost_expert_training': augmented_cost_expert_training,
                'cost_student_training': cost_student_training,
                'obst_avoidance_violation_student_training': obst_avoidance_violation_student_training,
                'dyn_lim_violation_student_training': dyn_lim_violation_student_training,
                'augmented_cost_student_training': augmented_cost_student_training,
                'epoch': epoch_idx
            })
            
            # terminate conditions
            # if epoch_counter is more than 1/5 of the total epoch and augmented cost of expert is less than augmented cost of student 5 times in a row, then stop training
            if epoch_counter >= num_epochs and np.mean(augmented_cost_expert_eval) < np.mean(augmented_cost_student_eval):
                overfitting_counter += 1
            else:
                overfitting_counter = 0
            if overfitting_counter >= 5:
                break
            
    return noise_pred_net

def main():

    "********************* ARGUMENTS *********************"

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirs', default='/media/jtorde/T7/gdp/evals-dir/evals4/tmp_dagger/2/demos/', help='directory', type=str)
    parser.add_argument('-s', '--save-dir', default='/media/jtorde/T7/gdp/models/', help='save directory', type=str)
    parser.add_argument('-g', '--gnn', default=True, help='use gnn', type=str2bool)
    parser.add_argument('-t', '--test', default=False, help='test (small dataset)', type=str2bool)
    args = parser.parse_args()

    "********************* PARAMETERS *********************"

    use_gnn = args.gnn                                                  # use gnn?
    is_test_run = args.test                                             # test run?
    dirs = args.dirs                                                    # list npz files in the directory
    save_dir = args.save_dir                                            # save directory
    device = th.device('cpu') if not use_gnn else th.device('cuda')     # if we use gnn, we need to use cpu (this is because we use shuffle and worker processes in dataloader)
    th.set_default_device(device)                                       # set default device                                   
    th.set_default_dtype(th.float32)                                    # set default dtype
    max_num_training_demos = 100000 if not is_test_run else 100         # max number of training demonstrations
    percentage_training = 0.8                                           # percentage of training demonstrations
    percentage_eval = 0.1                                               # percentage of evaluation demonstrations
    percentage_test = 0.1                                               # percentage of test demonstrations
    pred_horizon = 1                                                    # TODO remove (from diffuion policy paper)
    obs_horizon = 1                                                     # TODO remove (from diffuion policy paper)
    obs_dim = 43                                                        # if we use GNN, this will be overwritten in Conditional REsidualBlock1D and won't be used
    action_dim = 22                                                     # 15(pos) + 6(yaw) + 1(time)
    num_diffusion_iters = 100                                           # number of diffusion iterations
    num_epochs = 500                                                    # number of epochs
    num_eval = 10                                                       # number of evaluation data points
    batch_size = 32                                                     # batch size

    # create kwargs for dataset parameters
    kwargs = {
        'max_num_training_demos': max_num_training_demos,
        'percentage_training': percentage_training,
        'percentage_eval': percentage_eval,
        'percentage_test': percentage_test,
    }
    
    "********************* DATA *********************"

    # create dataset
    dataset_training, dataset_eval, dataset_test = create_gnn_dataset(dirs, device, **kwargs) if use_gnn else create_dataset(dirs, device, **kwargs)

    # create dataloader for training
    if use_gnn:
        # create dataloader for GNN
        dataloader_training = GNNDataLoader(
            dataset_training,
            batch_size=batch_size,                                      # if batch_size is less than 256, then CPU is faster than GPU on my computer
            shuffle=False if str(device)=='cuda' else True,             # shuffle True causes error Expected a 'cuda' str(device) type for generator but found 'cpu' https://github.com/dbolya/yolact/issues/664#issuecomment-878241658
            num_workers=0 if str(device)=='cuda' else 16 ,              # if we wanna use cuda, need to set num_workers=0
            pin_memory=False if str(device)=='cuda' else True,          # accelerate cpu-gpu transfer
            persistent_workers=False if str(device)=='cuda' else True,  # if we wanna use cuda, need to set False
        )
    else:
        dataloader_training = th.utils.data.DataLoader(
            dataset_training,
            batch_size=batch_size,
            num_workers=16,
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=True
        )

    "********************* NETWORK *********************"
    
    # create noise prediction network
    noise_pred_net, noise_scheduler = network_init(
        action_dim=action_dim,
        obs_dim=obs_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        num_diffusion_iters=num_diffusion_iters,
        dataset_training=dataset_training,
        use_gnn=use_gnn,
        device=device
    )

    "********************* TRAINING *********************"

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=noise_pred_net.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = th.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader_training) * num_epochs
    )

    # training loop
    noise_pred_net = train(
        num_epochs=num_epochs,
        dataloader_training=dataloader_training,
        dataset_training=dataset_training,
        device=device,
        noise_pred_net=noise_pred_net,
        noise_scheduler=noise_scheduler,
        ema=ema,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        use_gnn=use_gnn,
        dataset_eval=dataset_eval,
        pred_horizon=pred_horizon,
        num_diffusion_iters=num_diffusion_iters,
        action_dim=action_dim,
        save_dir=save_dir,
        num_eval=num_eval
    )

    # Weights of the EMA model
    # is used for inference
    ema_noise_pred_net = noise_pred_net
    ema.copy_to(ema_noise_pred_net.parameters())

    # save model
    th.save(ema_noise_pred_net.state_dict(), save_dir+'final_ema_noise_pred_net.pth')

    "********************* TEST *********************"

    # evaluate on test data
    cost_expert_test, obst_avoidance_violation_expert_test, dyn_lim_violation_expert_test, augmented_cost_expert_test, cost_student_test, obst_avoidance_violation_student_test, dyn_lim_violation_student_test, augmented_cost_student_test = evaluate(dataset_test, device, ema_noise_pred_net, use_gnn)

    # wandb logging
    wandb.log({
        'cost_expert_test': cost_expert_test,
        'obst_avoidance_violation_expert_test': obst_avoidance_violation_expert_test,
        'dyn_lim_violation_expert_test': dyn_lim_violation_expert_test,
        'augmented_cost_expert_test': augmented_cost_expert_test,
        'cost_student_test': cost_student_test,
        'obst_avoidance_violation_student_test': obst_avoidance_violation_student_test,
        'dyn_lim_violation_student_test': dyn_lim_violation_student_test,
        'augmented_cost_student_test': augmented_cost_student_test,
    })

    # print
    print("cost_expert_test: ", cost_expert_test)
    print("cost_student_test: ", cost_student_test)
    print("augmented_cost_expert_test: ", augmented_cost_expert_test)
    print("augmented_cost_student_test: ", augmented_cost_student_test)

if __name__ == '__main__':
    main()