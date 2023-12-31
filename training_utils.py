#!/usr/bin/env python3

# diffusion policy import
import numpy as np
import torch as th
import torch.nn as nn
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

# import
import torch as th

# network utils import
from utils import get_nactions, calculate_deep_panther_loss

# wanb
import wandb

# import from py_panther
from compression.utils.other import CostComputer

def evaluate_mlp_or_lstm(dataset, policy, num_eval=10):

    """
    Evaluate noise_pred_net
    @param dataset_eval: evaluation dataset
    @param noise_pred_net: noise prediction network
    """

    # get cost computer
    cost_computer = CostComputer()

    # num_eval
    num_eval = min(num_eval, len(dataset))

    cost_expert, obst_avoidance_violation_expert, dyn_lim_violation_expert, augmented_cost_expert = [], [], [], []
    cost_student, obst_avoidance_violation_student, dyn_lim_violation_student, augmented_cost_student = [], [], [], []

    for dataset_idx in range(num_eval):

        # get expert actions and student actions
        nob = dataset[dataset_idx][0].reshape(1, 1, -1)
        expert_action = dataset[dataset_idx][1]
        student_action = policy(nob)

        # reshape to make it compatible with cost_computer
        nob = nob.reshape(1, -1)
        student_action = student_action.reshape(student_action.shape[1], -1)

        # move to numpy
        nob = nob.detach().cpu().numpy()
        expert_action = expert_action.detach().cpu().numpy()
        student_action = student_action.detach().cpu().numpy()

        # compute cost for expert
        for j in range(expert_action.shape[0]): # expert_action.shape[1] is num_trajs
            cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = cost_computer.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(nob, expert_action[[j], :])
            cost_expert.append(cost)
            obst_avoidance_violation_expert.append(obst_avoidance_violation)
            dyn_lim_violation_expert.append(dyn_lim_violation)
            augmented_cost_expert.append(augmented_cost)

        # compute cost for student
        for j in range(student_action.shape[0]): # student_action.shape[1] is num_trajs
            cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = cost_computer.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(nob, student_action[[j], :])
            cost_student.append(cost)
            obst_avoidance_violation_student.append(obst_avoidance_violation)
            dyn_lim_violation_student.append(dyn_lim_violation)
            augmented_cost_student.append(augmented_cost)
        
    # convert to numpy
    cost_expert = np.array(cost_expert)
    obst_avoidance_violation_expert = np.array(obst_avoidance_violation_expert)
    dyn_lim_violation_expert = np.array(dyn_lim_violation_expert)
    augmented_cost_expert = np.array(augmented_cost_expert)
    cost_student = np.array(cost_student)
    obst_avoidance_violation_student = np.array(obst_avoidance_violation_student)
    dyn_lim_violation_student = np.array(dyn_lim_violation_student)
    augmented_cost_student = np.array(augmented_cost_student)

    # return
    return cost_expert, obst_avoidance_violation_expert, dyn_lim_violation_expert, augmented_cost_expert, cost_student, obst_avoidance_violation_student, dyn_lim_violation_student, augmented_cost_student

def evaluate_diffusion_model(dataset, device, noise_pred_net, noise_scheduler, num_trajs, num_diffusion_iters, action_dim, use_gnn, num_eval):

    """
    Evaluate noise_pred_net
    @param dataset_eval: evaluation dataset
    @param device: device to transfer data to
    @param noise_pred_net: noise prediction network
    """

    # get cost computer
    cost_computer = CostComputer()

    # get expert actions and student actions
    expert_actions, student_actions, nobs = get_nactions(noise_pred_net, noise_scheduler, dataset, num_trajs, num_diffusion_iters, action_dim, use_gnn, device, is_visualize=False, num_eval=num_eval)

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

        for j in range(num_trajs):
            
            cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = cost_computer.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(nob, student_action[j,:].reshape(1, -1))
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

def train_loop_diffusion_model(num_epochs, dataloader_training, dataset_training, device, noise_pred_net, noise_scheduler, ema, optimizer, lr_scheduler, use_gnn, dataset_eval, num_trajs, num_diffusion_iters, action_dim, save_dir, num_eval):

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
                    
                    if not use_gnn:
                        # (B, obs_horizon * obs_dim)
                        obs_cond = obs_cond.flatten(start_dim=1)
                        # (B, num_trajs, action_dim)
                        for j in range(num_trajs-1): # to replicate the expert action for num_trajs times
                            naction = th.cat((naction, naction[:, -1:, :]), dim=1)

                    # (B, num_trajs, action_dim)
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
                th.save(noise_pred_net.state_dict(), f'{save_dir}/diffusion_num_{epoch_counter}.pth')
                epoch_counter += 1
                    
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # each epoch, we evaluate the model on evaluation data
            costs = evaluate_diffusion_model(dataset_eval, device, noise_pred_net, noise_scheduler, num_trajs, num_diffusion_iters, action_dim, use_gnn, num_eval)
            # unpack
            cost_expert_eval, obst_avoidance_violation_expert_eval, dyn_lim_violation_expert_eval, augmented_cost_expert_eval, cost_student_eval, obst_avoidance_violation_student_eval, dyn_lim_violation_student_eval, augmented_cost_student_eval = costs

            # each epoch we evaluate the model on training data too (to check overfitting)
            costs = evaluate_diffusion_model(dataset_training, device, noise_pred_net, noise_scheduler, num_trajs, num_diffusion_iters, action_dim, use_gnn, num_eval)
            # unpack
            cost_expert_training, obst_avoidance_violation_expert_training, dyn_lim_violation_expert_training, augmented_cost_expert_training, cost_student_training, obst_avoidance_violation_student_training, dyn_lim_violation_student_training, augmented_cost_student_training = costs

            # wandb logging
            wandb.log({
                'min_cost_expert_eval': np.min(cost_expert_eval),
                'avg_cost_expert_eval': np.mean(cost_expert_eval),
                'min_cost_student_eval': np.min(cost_student_eval),
                'avg_cost_student_eval': np.mean(cost_student_eval),
                'min_cost_expert_training': np.min(cost_expert_training),
                'avg_cost_expert_training': np.mean(cost_expert_training),
                'min_cost_student_training': np.min(cost_student_training),
                'avg_cost_student_training': np.mean(cost_student_training),
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

def train_diffusion_model(num_epochs, dataloader_training, dataset_training, device, noise_pred_net, noise_scheduler, use_gnn, dataset_eval, num_trajs, num_diffusion_iters, action_dim, save_dir, num_eval):

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
    noise_pred_net = train_loop_diffusion_model(
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
        num_trajs=num_trajs,
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
    filename = f'{save_dir}/diffusion_model_final.pth' if not use_gnn else f'{save_dir}/gnn_diffusion_model_final.pth'
    th.save(ema_noise_pred_net.state_dict(), filename)

    return ema_noise_pred_net

def train_loop_mlp(num_epochs, dataloader_training, dataset_training, dataset_eval, action_dim, save_dir, policy, yaw_loss_weight, optimizer, lr_scheduler, num_eval):
    """
    Train MLP
    """

    # training loop
    wandb.init(project='mlp')
    epoch_counter = 0
    overfitting_counter = 0
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            
            # batch loop
            with tqdm(dataloader_training, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:

                    # calculate loss
                    loss = calculate_deep_panther_loss(nbatch, policy, yaw_loss_weight)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # logging
                    loss = loss.item()
                    epoch_loss.append(loss)
                    tepoch.set_postfix(loss=loss)

                    # wandb logging
                    wandb.log({'loss': loss, 'epoch': epoch_idx})

                # save model
                th.save(policy.state_dict(), f'{save_dir}/mlp_num_{epoch_counter}.pth')
                epoch_counter += 1
                    
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # each epoch, we evaluate the model on evaluation data
            with th.no_grad():
                costs = evaluate_mlp_or_lstm(dataset_eval, policy, num_eval)
            # unpack
            cost_expert_eval, obst_avoidance_violation_expert_eval, dyn_lim_violation_expert_eval, augmented_cost_expert_eval, cost_student_eval, obst_avoidance_violation_student_eval, dyn_lim_violation_student_eval, augmented_cost_student_eval = costs

            # each epoch we evaluate the model on training data too (to check overfitting)
            with th.no_grad():
                costs = evaluate_mlp_or_lstm(dataset_training, policy, num_eval)
            # unpack
            cost_expert_training, obst_avoidance_violation_expert_training, dyn_lim_violation_expert_training, augmented_cost_expert_training, cost_student_training, obst_avoidance_violation_student_training, dyn_lim_violation_student_training, augmented_cost_student_training = costs

            # wandb logging
            wandb.log({
                'min_cost_expert_eval': np.min(cost_expert_eval),
                'avg_cost_expert_eval': np.mean(cost_expert_eval),
                'min_cost_student_eval': np.min(cost_student_eval),
                'avg_cost_student_eval': np.mean(cost_student_eval),
                'min_cost_expert_training': np.min(cost_expert_training),
                'avg_cost_expert_training': np.mean(cost_expert_training),
                'min_cost_student_training': np.min(cost_student_training),
                'avg_cost_student_training': np.mean(cost_student_training),
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
            
    return policy

def train_mlp(num_epochs, dataloader_training, dataset_training, policy, dataset_eval, action_dim, save_dir, num_eval):

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = th.optim.AdamW(
        params=policy.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader_training) * num_epochs
    )

    # training loop
    policy = train_loop_mlp(
        num_epochs=num_epochs,
        dataloader_training=dataloader_training,
        dataset_training=dataset_training,
        dataset_eval=dataset_eval,
        action_dim=action_dim,
        save_dir=save_dir,
        policy=policy,
        yaw_loss_weight=1,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_eval=num_eval
    )

    # save model
    filename = f'{save_dir}/mlp_final.pth'
    th.save(policy.state_dict(), filename)

    return policy

def train_loop_lstm(num_epochs, dataloader_training, dataset_training, dataset_eval, action_dim, save_dir, policy, yaw_loss_weight, optimizer, lr_scheduler, num_eval):

    """
    Train LSTM
    """

    # training loop
    wandb.init(project='lstm')
    epoch_counter = 0
    overfitting_counter = 0
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            
            # batch loop
            with tqdm(dataloader_training, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:

                    # calculate loss
                    loss = calculate_deep_panther_loss(nbatch, policy, yaw_loss_weight, is_lstm=True)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # logging
                    loss = loss.item()
                    epoch_loss.append(loss)
                    tepoch.set_postfix(loss=loss)

                    # wandb logging
                    wandb.log({'loss': loss, 'epoch': epoch_idx})

                # save model
                th.save(policy.state_dict(), f'{save_dir}/lstm_num_{epoch_counter}.pth')
                epoch_counter += 1
                    
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # each epoch, we evaluate the model on evaluation data
            with th.no_grad():
                costs = evaluate_mlp_or_lstm(dataset_eval, policy, num_eval)
            # unpack
            cost_expert_eval, obst_avoidance_violation_expert_eval, dyn_lim_violation_expert_eval, augmented_cost_expert_eval, cost_student_eval, obst_avoidance_violation_student_eval, dyn_lim_violation_student_eval, augmented_cost_student_eval = costs

            # each epoch we evaluate the model on training data too (to check overfitting)
            with th.no_grad():
                costs = evaluate_mlp_or_lstm(dataset_training, policy, num_eval)
            # unpack
            cost_expert_training, obst_avoidance_violation_expert_training, dyn_lim_violation_expert_training, augmented_cost_expert_training, cost_student_training, obst_avoidance_violation_student_training, dyn_lim_violation_student_training, augmented_cost_student_training = costs

            # wandb logging
            wandb.log({
                'min_cost_expert_eval': np.min(cost_expert_eval),
                'avg_cost_expert_eval': np.mean(cost_expert_eval),
                'min_cost_student_eval': np.min(cost_student_eval),
                'avg_cost_student_eval': np.mean(cost_student_eval),
                'min_cost_expert_training': np.min(cost_expert_training),
                'avg_cost_expert_training': np.mean(cost_expert_training),
                'min_cost_student_training': np.min(cost_student_training),
                'avg_cost_student_training': np.mean(cost_student_training),
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
            
    return policy

def train_lstm(num_epochs, dataloader_training, dataset_training, policy, dataset_eval, action_dim, save_dir, num_eval):

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = th.optim.AdamW(
        params=policy.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader_training) * num_epochs
    )

    # training loop
    policy = train_loop_lstm(
        num_epochs=num_epochs,
        dataloader_training=dataloader_training,
        dataset_training=dataset_training,
        dataset_eval=dataset_eval,
        action_dim=action_dim,
        save_dir=save_dir,
        policy=policy,
        yaw_loss_weight=1,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_eval=num_eval
    )

    # save model
    filename = f'{save_dir}/lstm_final.pth'
    th.save(policy.state_dict(), filename)

    return policy

def test_net(dataset_test, device, policy, use_gnn, network_type):

    # evaluate on test data
    if network_type == 'diffusion':
        cost_expert_test, obst_avoidance_violation_expert_test, dyn_lim_violation_expert_test, augmented_cost_expert_test, cost_student_test, obst_avoidance_violation_student_test, dyn_lim_violation_student_test, augmented_cost_student_test = evaluate_diffusion_model(dataset_test, device, policy, use_gnn)
    elif network_type == 'mlp' or network_type == 'lstm':
        cost_expert_test, obst_avoidance_violation_expert_test, dyn_lim_violation_expert_test, augmented_cost_expert_test, cost_student_test, obst_avoidance_violation_student_test, dyn_lim_violation_student_test, augmented_cost_student_test = evaluate_mlp_or_lstm(dataset_test, policy, action_dim, num_eval=10)

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

def train_loop_transformer(num_epochs, dataloader_training, dataset_training, dataset_eval, action_dim, save_dir, policy, yaw_loss_weight, optimizer, lr_scheduler, num_eval):

    """
    Train Transformer
    """

    # training loop
    wandb.init(project='transformer')
    epoch_counter = 0
    overfitting_counter = 0
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            
            # batch loop
            with tqdm(dataloader_training, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:

                    # calculate loss
                    loss = calculate_deep_panther_loss(nbatch, policy, yaw_loss_weight, is_lstm=True)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # logging
                    loss = loss.item()
                    epoch_loss.append(loss)
                    tepoch.set_postfix(loss=loss)

                    # wandb logging
                    wandb.log({'loss': loss, 'epoch': epoch_idx})

                # save model
                th.save(policy.state_dict(), f'{save_dir}/transformer_num_{epoch_counter}.pth')
                epoch_counter += 1
                    
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # each epoch, we evaluate the model on evaluation data
            with th.no_grad():
                costs = evaluate_mlp_or_lstm(dataset_eval, policy, num_eval)
            # unpack
            cost_expert_eval, obst_avoidance_violation_expert_eval, dyn_lim_violation_expert_eval, augmented_cost_expert_eval, cost_student_eval, obst_avoidance_violation_student_eval, dyn_lim_violation_student_eval, augmented_cost_student_eval = costs

            # each epoch we evaluate the model on training data too (to check overfitting)
            with th.no_grad():
                costs = evaluate_mlp_or_lstm(dataset_training, policy, num_eval)
            # unpack
            cost_expert_training, obst_avoidance_violation_expert_training, dyn_lim_violation_expert_training, augmented_cost_expert_training, cost_student_training, obst_avoidance_violation_student_training, dyn_lim_violation_student_training, augmented_cost_student_training = costs

            # wandb logging
            wandb.log({
                'min_cost_expert_eval': np.min(cost_expert_eval),
                'avg_cost_expert_eval': np.mean(cost_expert_eval),
                'min_cost_student_eval': np.min(cost_student_eval),
                'avg_cost_student_eval': np.mean(cost_student_eval),
                'min_cost_expert_training': np.min(cost_expert_training),
                'avg_cost_expert_training': np.mean(cost_expert_training),
                'min_cost_student_training': np.min(cost_student_training),
                'avg_cost_student_training': np.mean(cost_student_training),
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
            
    return policy

def train_transformer(num_epochs, dataloader_training, dataset_training, policy, dataset_eval, action_dim, save_dir, num_eval):

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = th.optim.AdamW(
        params=policy.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader_training) * num_epochs
    )

    # training loop
    policy = train_loop_transformer(
        num_epochs=num_epochs,
        dataloader_training=dataloader_training,
        dataset_training=dataset_training,
        dataset_eval=dataset_eval,
        action_dim=action_dim,
        save_dir=save_dir,
        policy=policy,
        yaw_loss_weight=1,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_eval=num_eval
    )

    # save model
    filename = f'{save_dir}/transformer_final.pth'
    th.save(policy.state_dict(), filename)

    return policy
