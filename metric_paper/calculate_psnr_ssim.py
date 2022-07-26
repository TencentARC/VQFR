import argparse
import cv2
import glob
import numpy as np
import os
from tqdm import tqdm

from vqfr.metrics import calculate_psnr, calculate_ssim
from vqfr.utils.matlab_functions import bgr2ycbcr


def calculate_psnr_ssim(args):
    """Calculate PSNR and SSIM for images.
    """
    psnr_all = []
    ssim_all = []
    restored_list = sorted(glob.glob(os.path.join(args.restored_folder, '*')))
    gt_list = sorted(glob.glob(os.path.join(args.gt_folder, '*')))

    if args.test_y_channel:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for idx, (restored_path, gt_path) in enumerate(tqdm(zip(restored_list, gt_list))):
        img_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(restored_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if args.correct_mean_var:
            mean_l = []
            std_l = []
            for j in range(3):
                mean_l.append(np.mean(img_gt[:, :, j]))
                std_l.append(np.std(img_gt[:, :, j]))
            for j in range(3):
                # correct twice
                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

        if args.test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # calculate PSNR and SSIM
        psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        # print(f'{i+1:3d}: {basename:25}. \tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}')
        psnr_all.append(psnr)
        ssim_all.append(ssim)
    print(args.gt_folder)
    print(args.restored_folder)
    print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f} dB, SSIM: {sum(ssim_all) / len(ssim_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-restored_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument('-gt_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    parser.add_argument(
        '--test_y_channel',
        action='store_true',
        default=False,
        help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
    parser.add_argument(
        '--correct_mean_var', action='store_true', default=True, help='Correct the mean and var of restored images.')
    args = parser.parse_args()
    calculate_psnr_ssim(args)
