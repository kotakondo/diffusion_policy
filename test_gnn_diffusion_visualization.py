#!/usr/bin/env python3

# diffusion policy import
import torch as th

# env import
import os

# network utils import
from utils import create_dataset, str2bool, create_gnn_dataset, get_nactions, network_init
import os
import argparse

# network utils import
from torch_geometric.loader import DataLoader as GNNDataLoader


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
    max_num_training_demos = 10_000 if not is_test_run else 100         # max number of training demonstrations
    percentage_training = 0.8                                           # percentage of training demonstrations
    percentage_eval = 0.1                                               # percentage of evaluation demonstrations
    percentage_test = 0.1                                               # percentage of test demonstrations
    pred_horizon = 64                                                   # TODO remove (from diffuion policy paper)
    obs_horizon = 1                                                     # TODO remove (from diffuion policy paper)
    obs_dim = 43                                                        # if we use GNN, this will be overwritten in Conditional REsidualBlock1D and won't be used
    action_dim = 22                                                     # 15(pos) + 6(yaw) + 1(time)
    num_diffusion_iters = 100                                           # number of diffusion iterations
    num_epochs = 500                                                    # number of epochs
    num_eval = 10                                                       # number of evaluation data points
    batch_size = 128                                                    # batch size

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

    """ ********************* VISUALIZATION ********************* """

    # load model
    dir = str(save_dir)
    # get the latest model in the directory
    files = os.listdir(dir)
    files = [dir + '/' + file for file in files if file.endswith('.pth')]
    files.sort(key=os.path.getmtime)
    model_path = files[-1]
    print("model_path: ", model_path)
    model = th.load(model_path, map_location=device)
    noise_pred_net.load_state_dict(model)

    # choose random num_eval dataset
    dataset_training = th.utils.data.Subset(dataset_training, th.randperm(len(dataset_training))[:num_eval])

    # get expert actions and predicted actions(nactions)
    get_nactions(noise_pred_net, noise_scheduler, dataset_training, pred_horizon, num_diffusion_iters, action_dim, use_gnn, device, is_visualize=True, num_eval=10)


if __name__ == '__main__':
    main()