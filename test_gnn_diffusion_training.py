#!/usr/bin/env python3

# import
import torch as th
import argparse
import torch.nn as nn

# network utils import
from utils import create_dataset, str2bool, setup_diffusion, visualize
from network_utils import MLP
from torch_geometric.loader import DataLoader as GNNDataLoader

# training utils import
from training_utils import train_diffusion_model, train_non_diffusion_model, test_net

# wanb
import wandb
wandb.login()

# avoid using too much memory 
th.cuda.empty_cache()

# avoid opening too many files
th.multiprocessing.set_sharing_strategy('file_system')

def main():

    """ ********************* ARGUMENTS ********************* """

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--machine', default='kota2', help='machine', type=str)
    parser.add_argument('-d', '--data-dir', default='/media/jtorde/T7/gdp/evals-dir/evals4/tmp_dagger/2/demos/', help='directory', type=str)
    parser.add_argument('-s', '--save-dir', default='/media/jtorde/T7/gdp/models/', help='save directory', type=str)
    parser.add_argument('-t', '--test', default=False, help='test (small dataset)', type=str2bool)
    parser.add_argument('-en', '--en-network-type', default='mlp', help='encoder network type (/mlp/lstm/transformer/gnn)', type=str)
    parser.add_argument('-de', '--de-network-type', default='mlp', help='encoder network type (diffusion/mlp)', type=str)
    parser.add_argument('-o', '--observation-type', default='last', help='observation type (history/last)', type=str)
    parser.add_argument('-v', '--visualize', default=False, help='visualize', type=str2bool)
    parser.add_argument('--train-model', default=True, help='train model', type=str2bool)
    parser.add_argument('--model-path', default=None, help='model path', type=str)
    args = parser.parse_args()

    """ ********************* PARAMETERS ********************* """

    # parameters
    if args.machine == 'kota2':
        data_dir = args.data_dir
        save_dir = args.save_dir
    elif args.machine == 'lambda03' or args.machine == 'lambda04':
        data_dir = './evals-dir/evals4/tmp_dagger/2/demos/'
        save_dir = './models/'

    is_test_run = args.test                                             # test run?
    train_model = args.train_model                                      # train model?
    model_path = args.model_path                                        # model path
    en_network_type = args.en_network_type                              # encoder network type
    de_network_type = args.de_network_type                              # decoder network type
    obs_type = args.observation_type                                    # observation type
    is_visualize = args.visualize                                       # visualize?
    
    # default parameters
    device = th.device('cpu') #if not use_gnn else th.device('cuda')    # if we use gnn, we need to use cpu (this is because we use shuffle and worker processes in dataloader)
    th.set_default_device(device)                                       # set default device                                   
    th.set_default_dtype(th.float32)                                    # set default dtype
    max_num_training_demos = 10_000 if not is_test_run else 100         # max number of training demonstrations
    percentage_training = 0.8                                           # percentage of training demonstrations
    percentage_eval = 0.1                                               # percentage of evaluation demonstrations
    percentage_test = 0.1                                               # percentage of test demonstrations
    num_trajs = 8  # in diffusion model it has to be a multiple of 2    # num_trajs to predict
    obs_horizon = 1                                                     # TODO remove (from diffuion policy paper)
    obs_dim = 43                                                        # if we use GNN, this will be overwritten in Conditional REsidualBlock1D and won't be used
    action_dim = 22                                                     # 15(pos) + 6(yaw) + 1(time)
    num_diffusion_iters = 50                                            # number of diffusion iterations
    num_epochs = 500 if not is_test_run else 1                          # number of epochs
    num_eval = 10                                                       # number of evaluation data points
    batch_size = 128                                                    # batch size
    scheduler_type = 'ddim' # 'ddpm', 'ddim' or 'dpm-multistep'         # scheduler type (ddpm/dpm-multistep)
    yaw_loss_weight = 1.0                                               # yaw loss weight

    # check if we use GNN and last observation type
    if en_network_type == 'gnn' and obs_type != 'last':
        raise NotImplementedError("GNN only supports last observation type")

    """ ********************* NETWORK ********************* """
    # network parameters

    mlp_hidden_sizes = [1024, 1024]                                     # hidden sizes for mlp
    mlp_activation = nn.ReLU()                                          # activation for mlp
    lstm_hidden_size = 1024                                             # hidden size for lstm
    transformer_d_model = 43 if de_network_type == 'mlp' else 23        # d_model for transformer (43 for mlp, 1, 13, 23, 299 for diffusion)
    transformer_nhead = 43 if de_network_type == 'mlp' else 23          # nhead for transformer (43 for mlp, 1, 13, 23, 299 for diffusion)
    transformer_dim_feedforward = 1024                                  # feedforward_dim for transformer
    transformer_dropout = 0.1                                           # dropout for transformer
    gnn_hidden_channels = 1024                                          # hidden_channels for gnn
    gnn_num_layers = 4                                                  # num_layers for gnn
    gnn_num_heads = 4                                                   # num_heads for gnn
    gnn_group = 'max'                                                   # group for gnn

    # default model path
    # if args.model_path is not None:
    #     model_path = args.model_path
    # elif network_type == 'diffusion':
    #     model_path = '/media/jtorde/T7/gdp/models/model2/num_414_noise_pred_net.pth'
    #     num_trajs = 1 # when i trained the model, i used num_trajs=1
    # elif network_type == 'mlp':
    #     model_path = '/media/jtorde/T7/gdp/models/model7/mlp_final.pth'
    # elif network_type == 'lstm':
    #     model_path = '/media/jtorde/T7/gdp/models/model8/lstm_final.pth'
    # elif network_type == 'transformer':
    #     model_path = '/media/jtorde/T7/gdp/models/model1-lambda03/transformer_final.pth'

    """ ********************* DATA ********************* """

        # create kwargs for dataset parameters
    kwargs = {
        'data_dir': data_dir,
        'save_dir': save_dir,
        'is_test_run': is_test_run,
        'train_model': train_model,
        'en_network_type': en_network_type,
        'de_network_type': de_network_type,
        'obs_type': obs_type,
        'device': device,
        'max_num_training_demos': max_num_training_demos,
        'percentage_training': percentage_training,
        'percentage_eval': percentage_eval,
        'percentage_test': percentage_test,
        'num_trajs': num_trajs,
        'obs_horizon': obs_horizon,
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'num_diffusion_iters': num_diffusion_iters,
        'num_epochs': num_epochs,
        'num_eval': num_eval,
        'batch_size': batch_size,
        'scheduler_type': scheduler_type,
        'yaw_loss_weight': yaw_loss_weight,
        'model_path': model_path,
        'num_trajs': num_trajs,
        'input_dim': obs_dim,
        'output_dim': action_dim,
        'en_network_type': en_network_type,
        'mlp_hidden_sizes': mlp_hidden_sizes,
        'mlp_activation': mlp_activation,
        'lstm_hidden_size': lstm_hidden_size,
        'transformer_d_model': transformer_d_model,
        'transformer_nhead': transformer_nhead,
        'transformer_dim_feedforward': transformer_dim_feedforward,
        'transformer_dropout': transformer_dropout,
        'gnn_hidden_channels': gnn_hidden_channels,
        'gnn_num_layers': gnn_num_layers,
        'gnn_num_heads': gnn_num_heads,
        'gnn_group': gnn_group,
        'diffusion_step_embed_dim': 256,
        'diffusion_down_dims': [256, 512, 1024],
        'diffusion_kernel_size': 5,
        'diffusion_n_groups': 8,
        'machine': args.machine,
    }

    # create dataset
    dataset_training, dataset_eval, dataset_test = create_dataset(**kwargs)

    # we need gnn_data model to set up GNN model
    kwargs['gnn_data'] = dataset_training[0]

    # create dataloader for training
    if en_network_type == 'gnn':
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

    # create datasets_loader for data parameters
    datasets_loader = {
        'dataset_training': dataset_training,
        'dataset_eval': dataset_eval,
        'dataset_test': dataset_test,
        'dataloader_training': dataloader_training,
    }

    kwargs['datasets_loader'] = datasets_loader

    """ ********************* NETWORK ********************* """

    if de_network_type == 'diffusion':
        # create noise prediction network for diffusion model
        policy, noise_scheduler = setup_diffusion(**kwargs)

    elif de_network_type == 'mlp':
        # create policy for mlp
        policy = MLP(**kwargs)

    else:
        raise NotImplementedError
    
    noise_scheduler = None if de_network_type != 'diffusion' else noise_scheduler


    """ ********************* LOAD MODEL ********************* """

    if not train_model:
        # load model
        print("model_path: ", model_path)
        model = th.load(model_path, map_location=device)
        policy.load_state_dict(model)

    """ ********************* TRAINING ********************* """

    if train_model:
        if de_network_type == 'diffusion':
            # train diffusion model
            policy = train_diffusion_model(policy, noise_scheduler, **kwargs)
        elif de_network_type == 'mlp':
            # train mlp
            policy = train_non_diffusion_model(policy, **kwargs)
        else:
            raise NotImplementedError
    
    """ ********************* VISUALIZATION ********************* """

    if is_visualize:
        # TODO only supports diffusion model
        visualize(save_dir, use_gnn, device, num_eval, num_trajs, num_diffusion_iters, action_dim, policy, noise_scheduler, dataset_test)

    """ ********************* TEST ********************* """

    # test_net(policy, dataset_test, noise_scheduler=noise_scheduler, **kwargs)

if __name__ == '__main__':
    main()