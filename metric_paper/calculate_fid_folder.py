import argparse
import math
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from vqfr.data import build_dataset
from vqfr.metrics.fid import calculate_fid, extract_inception_features, load_patched_inception_v3


def calculate_fid_folder(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # inception model
    inception = load_patched_inception_v3(device)
    print(len(os.listdir(args.restored_folder)))
    # create dataset
    opt = {}
    opt['name'] = 'SingleImageDataset'
    opt['type'] = 'SingleImageDataset'
    opt['dataroot_lq'] = args.restored_folder
    opt['io_backend'] = dict(type=args.backend)
    opt['mean'] = [0.5, 0.5, 0.5]
    opt['std'] = [0.5, 0.5, 0.5]
    dataset = build_dataset(opt)

    # create dataloader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=None,
        drop_last=False)
    args.num_sample = min(args.num_sample, len(dataset))
    total_batch = math.ceil(args.num_sample / args.batch_size)

    def data_generator(data_loader, total_batch):
        for idx, data in enumerate(data_loader):
            if idx >= total_batch:
                break
            else:
                yield data['lq']

    features = extract_inception_features(data_generator(data_loader, total_batch), inception, total_batch, device)
    features = features.numpy()
    total_len = features.shape[0]
    features = features[:args.num_sample]
    print(f'Extracted {total_len} features, use the first {features.shape[0]} features to calculate stats.')

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    # load the dataset stats
    stats = torch.load(args.fid_stats)
    real_mean = stats['mean']
    real_cov = stats['cov']

    # calculate FID metric
    fid = calculate_fid(sample_mean, sample_cov, real_mean, real_cov)
    print(args.restored_folder)
    print('fid:', fid)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-restored_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument(
        '--fid_stats',
        type=str,
        help='Path to the dataset fid statistics.',
        default='experiments/pretrained_models/metric_weights/inception_FFHQ_512.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_sample', type=int, default=3000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--backend', type=str, default='disk', help='io backend for dataset. Option: disk, lmdb')
    args = parser.parse_args()
    calculate_fid_folder(args)
