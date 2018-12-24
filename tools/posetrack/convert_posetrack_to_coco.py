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
        default='/data/PoseTrack/posetrack_data/coco_annotations', type=str)
    parser.add_argument(
        '--datadir', help="data dir for annotations to be converted",
        default='/data/PoseTrack/posetrack_data', type=str)

    return parser.parse_args()


def convert_posetrack(data_dir, out_dir):
    sets = ['train', 'val', 'test']
    categories = [
        {'id': 1, 'name': 'person'}
    ]
    json_name = 'posetrack_instances_%s.json'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for data_set in sets:
        print('Starting {}'.format(data_set))
        ann_dir = os.path.join(data_dir, 'annotations', data_set)
        raw_images_dict, raw_annotations_dict, _ = load_annotations(ann_dir)

        images = []
        annotations = []
        seq_image_wh = {}
        for image_id, raw_image in tqdm(raw_images_dict.items(), total=len(raw_images_dict)):
            seq_name = raw_image['seq_name']
        
            # load image_wh
            if seq_name in seq_image_wh:
                image_width, image_height = seq_image_wh[seq_name]
            else:
                _image_filename = os.path.join(data_dir, raw_image['file_name'])
                _image = cv2.imread(_image_filename)
                image_height, image_width = _image.shape[:2]
                seq_image_wh[seq_name] = (image_width, image_height)

            # images
            image = {
                'id': image_id,
                'file_name': raw_image['file_name'],
                'width': image_width,
                'height': image_height,
            }
            images.append(image)

            raw_annotations = raw_annotations_dict.get(image_id, [])
            for raw_ann in raw_annotations:
                if 'bbox' not in raw_ann:
                    continue
                x1, y1, w, h = raw_ann['bbox']
                x2 = x1 + w
                y2 = y1 + h
                ann = {
                    'id': raw_ann['id'],
                    'image_id': image_id,
                    'category_id': 1,  # raw_ann['category_id']
                    'iscrowd': 0,
                    'area': w * h,
                    'bbox': raw_ann['bbox'],
                    'segmentation': [[x1, y1, x1, y2, x2, y2, x2, y1]],
                }
                annotations.append(ann)

        ann_dict = {
            'categories': categories,
            'images': images,
            'annotations': annotations
        }

        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(out_dir, json_name % data_set), 'w') as outfile:
            outfile.write(json.dumps(ann_dict))


def load_annotations(anno_root):
    all_images_dict = {}
    all_annotations_dict = {}
    seq_categories_dict = {}
    for seq_anno_file in os.listdir(anno_root):
        seq_name = os.path.splitext(seq_anno_file)[0]
        with open(os.path.join(anno_root, seq_anno_file), 'r') as f:
            raw_annotation = json.load(f)
        images = raw_annotation['images']
        annotations = raw_annotation.get('annotations', [])

        # select images
        for image in images:
            image['seq_name'] = seq_name
            all_images_dict[image['id']] = image

        # index annotations by image_id
        for anno in annotations:
            image_id = anno['image_id']
            all_annotations_dict.setdefault(image_id, [])
            all_annotations_dict[image_id].append(anno)

        # save categories
        seq_categories_dict[seq_name] = raw_annotation['categories']

    return all_images_dict, all_annotations_dict, seq_categories_dict


if __name__ == '__main__':
    args = parse_args()
    convert_posetrack(args.datadir, args.outdir)
