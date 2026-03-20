import os
import argparse
import time
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import torch

from data_utils.MN40_hdf5_Dataloader import ModelNet40
from data_utils.ScanObjectNNDataLoader import ScanObjectNNDataLoader
from utils.utils import set_seed

from attacks import PointCloudAttack
from utils.set_distance import ChamferDistance, HausdorffDistance



def load_data(args):
    """Load the dataset from the given path.
    """
    print('Start Loading Dataset...')
    if args.dataset == 'ModelNet40':
        TEST_DATASET = ModelNet40(
            num_points=args.input_point_nums,
            partition='test',
            data_dir=args.data_path
        )
    elif args.dataset == 'ScanObjectNN':
        TEST_DATASET = ScanObjectNNDataLoader(
            root=args.data_path,
            split='test',
            bg=False,
        )
    else:
        raise NotImplementedError

    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print('Finish Loading Dataset...')
    return testDataLoader



def data_preprocess(data):
    """Preprocess the given data and label.
    """
    points, target = data

    points = points # [B, N, C]
    target = target[:, 0] # [B]

    points = points.cuda()
    target = target.cuda()

    return points, target


def main():
    # load data
    test_loader = load_data(args)

    num_class = 0
    if args.dataset == 'ModelNet40':
        num_class = 40
    elif args.dataset == 'ScanObjectNN':
        num_class = 15
    assert num_class != 0
    args.num_class = num_class

    # load model
    attack = PointCloudAttack(args)

    # start attack
    atk_success = 0
    avg_chamfer_dist = 0.
    avg_hausdorff_dist = 0.
    avg_time_cost = 0.
    avg_l2_dist = 0.
    chamfer_loss = ChamferDistance()
    hausdorff_loss = HausdorffDistance()

    count = 0

    for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        points, target = data_preprocess(data)
        target = target.long()

        t0 = time.perf_counter()
        adv_points, adv_target = attack.run(points, target)
        t1 = time.perf_counter()
        avg_time_cost += t1 - t0

        atk_success += 1 if adv_target != target else 0
        # modified point num count
        points = points[:,:,:3].data # P, [1, N, 3]
        pert_pos = torch.where(abs(adv_points-points).sum(2))
        count_map = torch.zeros_like(points.sum(2))
        count_map[pert_pos] = 1.

        avg_chamfer_dist += chamfer_loss(adv_points, points)
        avg_hausdorff_dist += hausdorff_loss(adv_points, points)
        avg_l2_dist += torch.sqrt(((adv_points - points)**2).sum(1).sum(1))
        

    atk_success /= batch_id + 1
    print('Attack success rate: ', atk_success)
    avg_time_cost /= batch_id + 1
    print('Average time cost: ', avg_time_cost)
    avg_chamfer_dist /= batch_id + 1
    print('Average Chamfer Dist:', avg_chamfer_dist.item())
    avg_hausdorff_dist /= batch_id + 1
    print('Average Hausdorff Dist:', avg_hausdorff_dist.item())
    avg_l2_dist /= batch_id + 1
    print('Average L2 Dist:', avg_l2_dist.item())




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WA/SubAttack')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N', 
                        help='input batch size for training (default: 1)')
    parser.add_argument('--input_point_nums', type=int, default=1024,
                        help='Point nums of each point cloud')
    parser.add_argument('--seed', type=int, default=2022, metavar='S',
                        help='random seed (default: 2022)')
    parser.add_argument('--dataset', type=str, default='ModelNet40',
                        choices=['ModelNet40', 'ScanObjectNN'])
    parser.add_argument('--data_path', type=str, 
                        default=None, required=True, help='The path of the dataset')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Worker nums of data loading.')

    parser.add_argument('--transfer_attack_method', type=str, default=None,
                        choices=['WAAttack', 'SubAttack'])
    parser.add_argument('--surrogate_model', type=str, default='pointnet_cls',
                        choices=['pointnet_cls', 'dgcnn', 'pct', 'curvenet'])
    parser.add_argument('--target_model', type=str, default='pointnet_cls',
                        choices=['pointnet_cls', 'dgcnn', 'pct', 'curvenet'])
    parser.add_argument('--defense_method', type=str, default=None,
                        choices=['sor', 'srs', 'dupnet'])
    parser.add_argument('--top5_attack', action='store_true', default=False,
                        help='Whether to attack the top-5 prediction [default: False]')

    parser.add_argument('--max_steps', default=50, type=int,
                        help='max iterations for black-box attack')
    parser.add_argument('--eps', default=0.16, type=float,
                        help='epsilon of perturbation')
    parser.add_argument('--step_size', default=0.007, type=float,
                        help='step-size of perturbation')
    args = parser.parse_args()

    # basic configuration
    set_seed(args.seed)
    args.device = torch.device("cuda")

    # main loop
    main()
