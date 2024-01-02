#!/usr/bin/env python3

# diffusion policy import
import numpy as np
import torch as th
import os
import torch.nn as nn
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import datetime
# import
import torch as th

# network utils import
from utils import get_nactions, calculate_deep_panther_loss

# wanb
import wandb

# import from py_panther
# from compression.utils.other import CostComputer

def evaluate_non_diffusion_model(dataset, policy, **kwargs):

    """
    Evaluate noise_pred_net
    @param dataset_eval: evaluation dataset
    @param noise_pred_net: noise prediction network
    """

    # unpack
    num_eval = kwargs['num_eval']
    en_network_type = kwargs['en_network_type']

    # set policy to eval mode
    policy.eval()

    # get cost computer
    cost_computer = CostComputer()

    # num_eval
    num_eval = min(num_eval, len(dataset))

    cost_expert, obst_avoidance_violation_expert, dyn_lim_violation_expert, augmented_cost_expert = [], [], [], []
    cost_student, obst_avoidance_violation_student, dyn_lim_violation_student, augmented_cost_student = [], [], [], []
    for dataset_idx in range(num_eval):

        # get expert actions and student actions
        nob = dataset[dataset_idx]['obs']
        expert_action = dataset[dataset_idx]['acts']

        if en_network_type == 'gnn':
            x_dict = dataset[dataset_idx].x_dict
            edge_index_dict = dataset[dataset_idx].edge_index_dict
            student_action = policy(nob, x_dict, edge_index_dict) 
            nob = nob.squeeze(0) # remove the first dimension
            expert_action = expert_action.squeeze(0) # remove the first dimension
        else:
            student_action = policy(nob)

        student_action = student_action.squeeze(0) # remove the first dimension

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

def evaluate_diffusion_model(dataset, policy, noise_scheduler, **kwargs):

    """
    Evaluate policy
    @param dataset_eval: evaluation dataset
    @param device: device to transfer data to
    @param policy: noise prediction network
    """

    # get cost computer
    cost_computer = CostComputer()

    # get expert actions and student actions
    expert_actions, student_actions, nobs = get_nactions(policy, noise_scheduler, dataset, is_visualize=False, **kwargs)


    # evaluation for expert actions
    cost_expert, obst_avoidance_violation_expert, dyn_lim_violation_expert, augmented_cost_expert = [], [], [], []
    for nob, expert_action in zip(nobs, expert_actions):
        expert_action = expert_action.squeeze(0) # remove the first dimension
        nob = nob.squeeze(0) # remove the first dimension
        print("expert_action.shape: ", expert_action.shape)
        print("nob.shape: ", nob.shape)
        for j in range(expert_action.shape[0]): # for num_trajs we loop through
            # compute cost
            cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = cost_computer.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(nob, expert_action[[j], :])
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
        nob = nob.squeeze(0) # remove the first dimension
        print("student_action.shape: ", student_action.shape)
        print("nob.shape: ", nob.shape)
        for j in range(student_action.shape[0]): # for num_trajs we loop through
            # compute cost
            cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = cost_computer.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(nob, student_action[[j], :])
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

def train_loop_diffusion_model(policy, optimizer, lr_scheduler, noise_scheduler, ema, **kwargs):

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

    # unpack
    num_epochs = kwargs['num_epochs']
    dataloader_training = kwargs['datasets_loader']['dataloader_training']
    use_gnn = kwargs['en_network_type'] == 'gnn'
    device = kwargs['device']
    save_dir = kwargs['save_dir']
    dataset_eval = kwargs['datasets_loader']['dataset_eval']
    dataset_training = kwargs['datasets_loader']['dataset_training']
    en_network_type = kwargs['en_network_type']
    de_network_type = kwargs['de_network_type']

    # set policy to train mode
    policy.train()

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
                    nobs = nbatch['obs']
                    naction = nbatch['acts']

                    x_dict = nbatch.x_dict if use_gnn else None
                    edge_index_dict = nbatch.edge_index_dict if use_gnn else None
                    B = nobs.shape[0]

                    # observation as FiLM conditioning
                    # (B, obs_horizon, obs_dim)
                    obs_cond = nobs[:, :, :]
                    
                    # (B, obs_horizon * obs_dim)
                    obs_cond = obs_cond.flatten(start_dim=1)

                    # naction's num_trajs needs to be a multiple of 2 so let's make it 8 for now
                    # naction = naction[:, :8, :] # (B, num_trajs, action_dim)

                    # sample noise to add to actions
                    noise = th.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = th.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
                    
                    # predict the noise residual
                    noise_pred = policy(sample=noisy_actions, timestep=timesteps, global_cond=obs_cond, x_dict=x_dict, edge_index_dict=edge_index_dict)

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
                    ema.step(policy.parameters())

                    # logging
                    loss = loss.item()
                    epoch_loss.append(loss)
                    tepoch.set_postfix(loss=loss)

                    # wandb logging
                    wandb.log({'loss': loss, 'epoch': epoch_idx})

                # save model
                filename = f'{save_dir}/{en_network_type}_{de_network_type}_num_{epoch_counter}.pth'
                th.save(policy.state_dict(), filename)
                epoch_counter += 1
                    
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # # each epoch, we evaluate the model on evaluation data
            # costs = evaluate_diffusion_model(dataset_eval, policy, noise_scheduler, **kwargs)
            # # unpack
            # cost_expert_eval, obst_avoidance_violation_expert_eval, dyn_lim_violation_expert_eval, augmented_cost_expert_eval, cost_student_eval, obst_avoidance_violation_student_eval, dyn_lim_violation_student_eval, augmented_cost_student_eval = costs

            # # each epoch we evaluate the model on training data too (to check overfitting)
            # costs = evaluate_diffusion_model(dataset_training, policy, noise_scheduler, **kwargs)
            # # unpack
            # cost_expert_training, obst_avoidance_violation_expert_training, dyn_lim_violation_expert_training, augmented_cost_expert_training, cost_student_training, obst_avoidance_violation_student_training, dyn_lim_violation_student_training, augmented_cost_student_training = costs

            # # wandb logging
            # wandb.log({
            #     'min_cost_expert_eval': np.min(cost_expert_eval),
            #     'avg_cost_expert_eval': np.mean(cost_expert_eval),
            #     'min_cost_student_eval': np.min(cost_student_eval),
            #     'avg_cost_student_eval': np.mean(cost_student_eval),
            #     'min_cost_expert_training': np.min(cost_expert_training),
            #     'avg_cost_expert_training': np.mean(cost_expert_training),
            #     'min_cost_student_training': np.min(cost_student_training),
            #     'avg_cost_student_training': np.mean(cost_student_training),
            #     'epoch': epoch_idx
            # })
            
            # # terminate conditions
            # # if epoch_counter is more than 1/5 of the total epoch and augmented cost of expert is less than augmented cost of student 5 times in a row, then stop training
            # if epoch_counter >= num_epochs and np.mean(augmented_cost_expert_eval) < np.mean(augmented_cost_student_eval):
            #     overfitting_counter += 1
            # else:
            #     overfitting_counter = 0
            # if overfitting_counter >= 5:
            #     break
            
    return policy

def train_diffusion_model(policy, noise_scheduler, **kwargs):

    """
    Train diffusion model
    """

    # unpack
    dataloader_training = kwargs['datasets_loader']['dataloader_training']
    num_epochs = kwargs['num_epochs']
    save_dir = kwargs['save_dir']
    use_gnn = kwargs['en_network_type'] == 'gnn'
    en_network_type = kwargs['en_network_type']
    de_network_type = kwargs['de_network_type']

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=policy.parameters(),
        power=0.75)

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
    policy = train_loop_diffusion_model(policy, optimizer, lr_scheduler, noise_scheduler, ema, **kwargs) 

    # Weights of the EMA model
    # is used for inference
    ema.copy_to(policy.parameters())

    # save model
    filename = f'{save_dir}/{en_network_type}_{de_network_type}_final.pth'
    th.save(policy.state_dict(), filename)

    return policy

def train_loop_non_diffusion_model(policy, optimizer, lr_scheduler, **kwargs):
    """
    Train MLP/LSTM/Transformer
    """

    # unpack
    num_epochs = kwargs['num_epochs']
    dataloader_training = kwargs['datasets_loader']['dataloader_training']
    dataset_training = kwargs['datasets_loader']['dataset_training']
    dataset_eval = kwargs['datasets_loader']['dataset_eval']
    save_dir = kwargs['save_dir']
    num_eval = kwargs['num_eval']
    en_network_type = kwargs['en_network_type']
    de_network_type = kwargs['de_network_type']

    # training loop
    wandb.init(project=en_network_type)
    epoch_counter = 0
    overfitting_counter = 0
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            
            # batch loop
            with tqdm(dataloader_training, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:

                    loss = calculate_deep_panther_loss(nbatch, policy, **kwargs) # calculate loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                    # logging
                    loss = loss.item()
                    epoch_loss.append(loss)
                    tepoch.set_postfix(loss=loss)

                    # wandb logging
                    wandb.log({'loss': loss, 'epoch': epoch_idx})

                # save model
                filename = f'{save_dir}/{en_network_type}_{de_network_type}_num_{epoch_counter}.pth'
                th.save(policy.state_dict(), filename)
                epoch_counter += 1
                    
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # each epoch, we evaluate the model on evaluation data
            # with th.no_grad():
            #     costs = evaluate_non_diffusion_model(dataset_eval, policy, **kwargs)
            # # unpack
            # cost_expert_eval, obst_avoidance_violation_expert_eval, dyn_lim_violation_expert_eval, augmented_cost_expert_eval, cost_student_eval, obst_avoidance_violation_student_eval, dyn_lim_violation_student_eval, augmented_cost_student_eval = costs

            # # each epoch we evaluate the model on training data too (to check overfitting)
            # with th.no_grad():
            #     costs = evaluate_non_diffusion_model(dataset_eval, policy, **kwargs)
            # # unpack
            # cost_expert_training, obst_avoidance_violation_expert_training, dyn_lim_violation_expert_training, augmented_cost_expert_training, cost_student_training, obst_avoidance_violation_student_training, dyn_lim_violation_student_training, augmented_cost_student_training = costs

            # # wandb logging
            # wandb.log({
            #     'min_cost_expert_eval': np.min(cost_expert_eval),
            #     'avg_cost_expert_eval': np.mean(cost_expert_eval),
            #     'min_cost_student_eval': np.min(cost_student_eval),
            #     'avg_cost_student_eval': np.mean(cost_student_eval),
            #     'min_cost_expert_training': np.min(cost_expert_training),
            #     'avg_cost_expert_training': np.mean(cost_expert_training),
            #     'min_cost_student_training': np.min(cost_student_training),
            #     'avg_cost_student_training': np.mean(cost_student_training),
            #     'epoch': epoch_idx
            # })
            
            # # terminate conditions
            # # if epoch_counter is more than 1/5 of the total epoch and augmented cost of expert is less than augmented cost of student 5 times in a row, then stop training
            # if epoch_counter >= num_epochs and np.mean(augmented_cost_expert_eval) < np.mean(augmented_cost_student_eval):
            #     overfitting_counter += 1
            # else:
            #     overfitting_counter = 0
            # if overfitting_counter >= 5:
            #     break
            
    return policy

def train_non_diffusion_model(policy, **kwargs):

    """
    Train MLP
    """

    # unpack
    num_epochs = kwargs['num_epochs']
    dataloader_training = kwargs['datasets_loader']['dataloader_training']
    save_dir = kwargs['save_dir']
    en_network_type = kwargs['en_network_type']
    de_network_type = kwargs['de_network_type']

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
    policy = train_loop_non_diffusion_model(policy, optimizer, lr_scheduler, **kwargs)

    # save model
    filename = f'{save_dir}/{en_network_type}_{de_network_type}_final.pth'
    th.save(policy.state_dict(), filename)

    return policy

def test_net(policy, dataset, noise_scheduler=None, **kwargs):

    """
    Test policy after training
    """

    # unpack
    en_network_type = kwargs['en_network_type']
    de_network_type = kwargs['de_network_type']
    train_model = kwargs['train_model']
    save_dir = kwargs['save_dir']
    model_path = kwargs['model_path']

    # evaluate on test data
    if de_network_type == 'diffusion':
        costs = evaluate_diffusion_model(dataset, policy, noise_scheduler, **kwargs)
    elif de_network_type == 'mlp' or de_network_type == 'lstm' or de_network_type == 'transformer':
        costs = evaluate_non_diffusion_model(dataset, policy, **kwargs)
    cost_expert_test, obst_avoidance_violation_expert_test, dyn_lim_violation_expert_test, augmented_cost_expert_test, cost_student_test, obst_avoidance_violation_student_test, dyn_lim_violation_student_test, augmented_cost_student_test = costs


    # print
    print("en_network_type:             ", en_network_type)
    print("de_network_type:             ", de_network_type)
    print("cost_expert_test:            ", np.mean(cost_expert_test))
    print("cost_student_test:           ", np.mean(cost_student_test))
    print("augmented_cost_expert_test:  ", np.mean(augmented_cost_expert_test))
    print("augmented_cost_student_test: ", np.mean(augmented_cost_student_test))
    
    # wandb logging
    if train_model: # if we are training the model, then log the test results
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

    # save results in file
    path = save_dir if train_model else model_path
    with open(f'/media/jtorde/T7/gdp/benchmark_results.txt', 'a') as f:
        f.write(f'date:                        {datetime.datetime.now()}\n')
        f.write(f'en_network_type:             {en_network_type}\n')
        f.write(f'de_network_type:             {de_network_type}\n')
        f.write(f'model_path:                  {path}\n')
        f.write(f'cost_expert_test:            {np.mean(cost_expert_test)}\n')
        f.write(f'cost_student_test:           {np.mean(cost_student_test)}\n')
        f.write(f'augmented_cost_expert_test:  {np.mean(augmented_cost_expert_test)}\n')
        f.write(f'augmented_cost_student_test: {np.mean(augmented_cost_student_test)}\n')
        f.write(f'\n')