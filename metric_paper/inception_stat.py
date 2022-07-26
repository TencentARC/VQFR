import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from vqfr.data import build_dataset
from vqfr.metrics.fid import extract_inception_features, load_patched_inception_v3


def inception_fid_stat_folder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='Path to the folder.', default='datasets/ffhq/ffhq_256/')
    parser.add_argument(
        '--fid_stats', type=str, help='Path to the save datasets fid statistics.', default='inception_FFHQ_256.pth')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_sample', type=int, default=100000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--backend', type=str, default='disk', help='io backend for dataset. Option: disk, lmdb')
    args = parser.parse_args()

    # inception model
    inception = load_patched_inception_v3(device)
    import os
    print(len(os.listdir(args.folder)))
    # create dataset
    opt = {}
    opt['name'] = 'SingleImageDataset'
    opt['type'] = 'SingleImageDataset'
    opt['dataroot_lq'] = args.folder
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
    state_dict = {'name': 'FFHQ', 'size': 256, 'mean': sample_mean, 'cov': sample_cov}
    torch.save(state_dict, args.fid_stats)


if __name__ == '__main__':
    inception_fid_stat_folder()
