import argparse
import cv2
import os
import warnings
from tqdm import tqdm

from vqfr.metrics import calculate_niqe
from vqfr.utils import scandir


def calculate_niqe_folder(args):

    niqe_all = []
    img_list = sorted(scandir(args.restored_folder, recursive=True, full_path=True))

    for i, img_path in enumerate(tqdm(img_list)):
        basename, _ = os.path.splitext(os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            niqe_score = calculate_niqe(img, args.crop_border, input_order='HWC', convert_to='y')
        # print(f'{i+1:3d}: {basename:25}. \tNIQE: {niqe_score:.6f}')
        niqe_all.append(niqe_score)

    print(args.restored_folder)
    print(f'Average: NIQE: {sum(niqe_all) / len(niqe_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-restored_folder', type=str, required=True, help='Path to the folder.')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    args = parser.parse_args()
    calculate_niqe_folder(args)
