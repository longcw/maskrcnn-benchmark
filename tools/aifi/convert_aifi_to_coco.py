from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import ujson as json
import os
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--outdir', help="output dir for json files",
        default='/media/longc/LSSD/Public/PILSNU/coco_annotations', type=str)
    parser.add_argument(
        '--datadir', help="data dir for annotations to be converted",
        default='/media/longc/LSSD/Public/PILSNU', type=str)

    return parser.parse_args()


def convert_tracking(data_dir, out_dir):
    categories = [
        {'id': 1, 'name': 'person'}
    ]
    json_name = 'image_names.json'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('Processing {}'.format(data_dir))
    image_root = os.path.join(data_dir, 'frames')
    image_filenames = load_images(image_root)

    images = []
    annotations = []
    image_wh = None
    for i, image_filename in tqdm(enumerate(image_filenames), total=len(image_filenames)):
        if image_wh is None:
            image = cv2.imread(os.path.join(image_root, image_filename))
            image_wh = (image.shape[1], image.shape[0])

        image_id = i + 1

        # images
        image = {
            'id': image_id,
            'file_name': image_filename,
            'width': image_wh[0],
            'height': image_wh[1],
        }
        images.append(image)

    ann_dict = {
        'categories': categories,
        'images': images,
        'annotations': annotations
    }

    print("Num categories: %s" % len(categories))
    print("Num images: %s" % len(images))
    print("Num annotations: %s" % len(annotations))
    with open(os.path.join(out_dir, json_name), 'w') as outfile:
        outfile.write(json.dumps(ann_dict))


def load_images(image_root):
    image_filenames = []
    curr_dir = os.path.abspath(os.curdir)
    os.chdir(image_root)
    for root, dirs, filenames in os.walk('.'):
        for filename in sorted(filenames):
            name, ext = os.path.splitext(filename)
            if ext in {'.jpg', '.png'}:
                image_filenames.append(os.path.join(root, filename))
    os.chdir(curr_dir)
    return image_filenames


if __name__ == '__main__':
    args = parse_args()
    convert_tracking(args.datadir, args.outdir)
