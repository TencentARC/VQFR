import argparse
import cv2
import glob
import numpy as np
import os.path as osp
from tqdm import tqdm

from vqfr.archs import init_alignment_model


def get_landmark_distance(gt_landmark, pred_landmark):
    return np.sqrt(((gt_landmark - pred_landmark)**2).sum(1)).mean()


def calculate_landmark_distance(args):

    # Configurations
    # -------------------------------------------------------------------------
    landmark_detector = init_alignment_model().cuda()  # RGB, normalized to [-1,1]
    landmark_l2_all = []

    img_list = sorted(glob.glob(osp.join(args.gt_folder, '*')))
    restored_list = sorted(glob.glob(osp.join(args.restored_folder, '*')))

    for i, (restored_path, img_path) in enumerate(tqdm(zip(restored_list, img_list))):
        img_gt = cv2.imread(img_path)
        img_restored = cv2.imread(restored_path)

        pred_landmark = landmark_detector.get_landmarks(img_restored)
        gt_landmark = landmark_detector.get_landmarks(img_gt)
        landmark_l2_all.append(get_landmark_distance(gt_landmark, pred_landmark))

    print(args.restored_folder)
    print(f'Average: Landmark L2: {sum(landmark_l2_all) / len(landmark_l2_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-restored_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument('-gt_folder', type=str, help='Path to the folder.', required=True)
    args = parser.parse_args()
    calculate_landmark_distance(args)
