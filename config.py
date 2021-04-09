##################################################
# Imports
##################################################

import argparse

# Custom
from splits import get_splits


def parse_args():

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_path', type=str, default='./data', help='The data path.')
    parser.add_argument('--val_ratio', type=float, default=0.2, 
                        help='The validation ratio used for the train/validation split.')
    parser.add_argument('--seed', type=int, default=1234, help='The random seed, used for reproducibility.')
    parser.add_argument('--known_classes', type=str, default='', 
                        help='The string known classes. Each class is specified with an integer and the classes are separated with spaces (for example "0 1 2").')
    parser.add_argument('--unknown_classes', type=str, default='', 
                        help='The string unknown classes. Each class is specified with an integer and the classes are separated with spaces (for example "5 7 8").')
    parser.add_argument('--split_num', type=int, default=0, 
                        help='The number of the data split associated with the splits in the splits.py file.')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for the dataloader.')
    parser.add_argument('--dataset', type=str, default='mnist', help='The dataset used.')

    # Model args
    parser.add_argument('--z_dim', type=int, default=128, help='Dimension of the latent capsule.')
    parser.add_argument('--lr', type=float, default=5e-5, help='The learning rate.')
    parser.add_argument('--t_mu_shift', type=float, default=1.0, help='Shift initialization of the targets.')
    parser.add_argument('--t_var_scale', type=float, default=1.0, help='Scale initialization of the targets.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for the contrastive term in the loss function.')
    parser.add_argument('--beta', type=float, default=0.01, help='Weight for the reconstruction term in the loss function.')
    parser.add_argument('--margin', type=float, default=10.0, help='Margin for the contrastive term in the loss function.')
    parser.add_argument('--in_dim_caps', type=int, default=16, help='Dimension of the input capsule.')
    parser.add_argument('--out_dim_caps', type=int, default=32, help='Dimension of the output capsule.')
    parser.add_argument('--checkpoint', type=str, default='', help='Model checkpoint path.')

    # Main
    parser.add_argument('--mode', type=str, default='train', help='Mode of the code. Can be set to "train" or "test".')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs.')

    # Parse
    args = parser.parse_args()

    # Process args
    known_classes = [int(cl) for cl in args.known_classes.split()]
    unknown_classes = [int(cl) for cl in args.unknown_classes.split()]
    splits = get_splits(args.dataset, num_split=args.split_num)
    args.known_classes = known_classes if len(known_classes) > 0 else splits['known_classes']
    args.unknown_classes = unknown_classes if len(unknown_classes) > 0 else splits['unknown_classes']
    return args
