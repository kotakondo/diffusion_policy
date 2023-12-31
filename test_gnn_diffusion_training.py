#!/usr/bin/env python3

# import
import torch as th
import argparse

# network utils import
from utils import create_dataset, create_gnn_dataset, str2bool, network_init
from network_utils import MLP, LSTM, Transformer
from torch_geometric.loader import DataLoader as GNNDataLoader

# training utils import
from training_utils import train_diffusion_model, train_mlp, train_lstm, train_transformer, test_net

# wanb
import wandb
wandb.login()

# avoid opening too many files
th.multiprocessing.set_sharing_strategy('file_system')

def main():

    "********************* ARGUMENTS *********************"

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirs', default='/media/jtorde/T7/gdp/evals-dir/evals4/tmp_dagger/2/demos/', help='directory', type=str)
    parser.add_argument('-s', '--save-dir', default='/media/jtorde/T7/gdp/models/', help='save directory', type=str)
    parser.add_argument('-g', '--gnn', default=True, help='use gnn', type=str2bool)
    parser.add_argument('-t', '--test', default=False, help='test (small dataset)', type=str2bool)
    parser.add_argument('-n', '--network-type', default='diffusion', help='network type (diffusion/mlp/lstm/transformer)', type=str)
    args = parser.parse_args()

    "********************* PARAMETERS *********************"

    use_gnn = args.gnn                                                  # use gnn?
    is_test_run = args.test                                             # test run?
    dirs = args.dirs                                                    # list npz files in the directory
    save_dir = args.save_dir                                            # save directory
    device = th.device('cuda') #if not use_gnn else th.device('cuda')     # if we use gnn, we need to use cpu (this is because we use shuffle and worker processes in dataloader)
    th.set_default_device(device)                                       # set default device                                   
    th.set_default_dtype(th.float32)                                    # set default dtype
    max_num_training_demos = 10_000 if not is_test_run else 100         # max number of training demonstrations
    percentage_training = 0.8                                           # percentage of training demonstrations
    percentage_eval = 0.1                                               # percentage of evaluation demonstrations
    percentage_test = 0.1                                               # percentage of test demonstrations
    num_trajs = 10  # in diffusion model it has to be a multiple of 2   # num_trajs to predict
    obs_horizon = 1                                                     # TODO remove (from diffuion policy paper)
    obs_dim = 43                                                        # if we use GNN, this will be overwritten in Conditional REsidualBlock1D and won't be used
    action_dim = 22                                                     # 15(pos) + 6(yaw) + 1(time)
    num_diffusion_iters = 100                                           # number of diffusion iterations
    num_epochs = 500                                                    # number of epochs
    num_eval = 10                                                       # number of evaluation data points
    batch_size = 128                                                    # batch size
    network_type = args.network_type                                    # network type

    # create kwargs for dataset parameters
    kwargs = {
        'max_num_training_demos': max_num_training_demos,
        'percentage_training': percentage_training,
        'percentage_eval': percentage_eval,
        'percentage_test': percentage_test,
        'network_type': network_type
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
    elif device == th.device('cpu'):
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
    else:
        dataloader_training = th.utils.data.DataLoader(
            dataset_training,
            batch_size=batch_size,
        )

    "********************* NETWORK *********************"

    if network_type == 'diffusion':
        # create noise prediction network for diffusion model
        noise_pred_net, noise_scheduler = network_init(
            action_dim=action_dim,
            obs_dim=obs_dim,
            obs_horizon=obs_horizon,
            num_trajs=num_trajs,
            num_diffusion_iters=num_diffusion_iters,
            dataset_training=dataset_training,
            use_gnn=use_gnn,
            device=device
        )

    elif network_type == 'mlp':
        # create policy for mlp
        policy = MLP(input_dim=43, output_dim=22, num_trajs=num_trajs, hidden_sizes=[1024, 1024])

    elif network_type == 'lstm':
        # create policy for lstm
        policy = LSTM(input_dim=43, output_dim=22, num_trajs=num_trajs, hidden_size=1024)

    elif network_type == 'transformer':
        # create policy for transformer
        policy = Transformer(input_dim=43, output_dim=22, num_trajs=num_trajs)
    else:
        raise NotImplementedError
    
    "********************* TRAINING *********************"

    if network_type == 'diffusion':
        # train diffusion model
        policy = train_diffusion_model(num_epochs, dataloader_training, dataset_training, device, noise_pred_net, noise_scheduler, use_gnn, dataset_eval, num_trajs, num_diffusion_iters, action_dim, save_dir, num_eval)
    elif network_type == 'mlp':
        # train mlp
        policy = train_mlp(num_epochs, dataloader_training, dataset_training, policy, dataset_eval, action_dim, save_dir, num_eval)
    elif network_type == 'lstm':
        # train lstm
        policy = train_lstm(num_epochs, dataloader_training, dataset_training, policy, dataset_eval, action_dim, save_dir, num_eval)
    elif network_type == 'transformer':
        # train transformer
        policy = train_transformer(num_epochs, dataloader_training, dataset_training, policy, dataset_eval, action_dim, save_dir, num_eval)
    else:
        raise NotImplementedError
    
    "********************* TEST *********************"

    test_net(dataset_test, device, policy, use_gnn, network_type)

if __name__ == '__main__':
    main()