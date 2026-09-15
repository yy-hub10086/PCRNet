# -*- encoding: utf-8 -*-
'''
@File    :   process_data.py
@Time    :   2024
@Description:   Generate occluded images for CUHK-PEDES dataset (independent version)
'''
import numpy as np
import math
from PIL import Image
import random
import os
import argparse
import json


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_txt(content, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(content)


def split_json(args):
    """
    Split dataset into train, test, and val sets
    """
    reid_raw_data = read_json(args.json_root)
    name = args.data_name
    train_json = []
    test_json = []
    val_json = []
    
    if name == 'ICFG-PEDES':
        for data in reid_raw_data:
            data_save = {
                'img_path': 'imgs/' + data['file_path'],
                'id': data['id'],
                'captions': data['captions']
            }
            split = data['split'].lower()
            if split == 'train':
                train_json.append(data_save)
            elif split == 'test' and (len(test_json) < 10000):
                test_json.append(data_save)
            else:
                val_json.append(data_save)
                
    elif name == 'CUHK-PEDES':
        for data in reid_raw_data:
            data_save = {
                'img_path': 'imgs/' + data['file_path'],
                'id': data['id'],
                'captions': data['captions']
            }
            
            split = data['split'].lower()
            if split == 'train':
                train_json.append(data_save)
            elif split == 'test':
                test_json.append(data_save)
            else:
                val_json.append(data_save)
                
    elif name == 'RSTPReid':
        for data in reid_raw_data:
            data_save = {
                'img_path': 'imgs/' + data['img_path'],
                'id': data['id'],
                'captions': data['captions']
            }

            split = data['split'].lower()
            if split == 'train':
                train_json.append(data_save)
            elif split == 'test':
                test_json.append(data_save)
            else:
                val_json.append(data_save)

    return train_json, test_json, val_json


def generate_caption(data_json, args, output_dir, dt_type):
    img_id_save = []
    caption_id_save = []
    img_path_save = []
    caption_save = []
    same_id_index_save = []
    data_save_by_id = {}
    data_save_by_id2 = {}
    dt_name = args.data_name
    
    for data in data_json:
        if dt_name == 'CUHK-PEDES':
            if data['id'] in [1369, 4116, 6116]:
                continue
            if data['id'] > 6116:
                id_new = data['id'] - 4
            elif data['id'] > 4116:
                id_new = data['id'] - 3
            elif data['id'] > 1369:
                id_new = data['id'] - 2
            else:
                id_new = data['id'] - 1
        else:
            id_new = data['id']

        data_save_i = {
            'img_path': data['img_path'],
            'id': id_new,
            'captions': data['captions']
        }
        if id_new not in data_save_by_id.keys():
            data_save_by_id[id_new] = []
            data_save_by_id2[id_new] = []
        data_save_by_id[id_new].append(data_save_i)
        indexx = data_json.index(data)
        data_save_by_id2[id_new].append(indexx)

    write_json(data_save_by_id2, os.path.join(output_dir, '{}_data_save_by_id'.format(dt_type)))
    data_order = 0
    print(dt_type)
    max_same_id = 0
    min_same_id = 28
    
    for id_new, data_save_by_id_i in data_save_by_id.items():
        caption_length = 0
        for data_save_by_id_i_i in data_save_by_id_i:
            caption_length += len(data_save_by_id_i_i['captions'])

        data_order_i = data_order + np.arange(caption_length)
        data_order_i_begin = 0

        for data_save_by_id_i_i in data_save_by_id_i:
            caption_length_i = len(data_save_by_id_i_i['captions'])
            data_order_i_end = data_order_i_begin + caption_length_i
            data_order_i_select = np.delete(data_order_i, np.arange(data_order_i_begin, data_order_i_end))
            data_order_i_begin = data_order_i_end
            
            if data_order_i_select.size > max_same_id:
                max_same_id = data_order_i_select.size
            if data_order_i_select.size < min_same_id:
                min_same_id = data_order_i_select.size
                
            if dt_type == "train":
                for j in range(caption_length_i):
                    img_id_save.append(data_save_by_id_i_i['id'])
                    caption_id_save.append(data_save_by_id_i_i['id'])
                    img_path_save.append(data_save_by_id_i_i['img_path'])
                    same_id_index_save.append(data_order_i_select.tolist())
                    caption_j = data_save_by_id_i_i['captions'][j]
                    caption_save.append(caption_j)
            else:
                img_id_save.append(data_save_by_id_i_i['id'])
                img_path_save.append(data_save_by_id_i_i['img_path'])
                for j in range(caption_length_i):
                    caption_id_save.append(data_save_by_id_i_i['id'])
                    caption_j = data_save_by_id_i_i['captions'][j]
                    caption_save.append(caption_j)

        data_order = data_order + caption_length

    data_save = {
        'img_id': img_id_save,
        'caption_id': caption_id_save,
        'img_path': img_path_save,
        'max_same_id': max_same_id,
        'min_same_id': min_same_id,
        'same_id_index': same_id_index_save,
        'captions': caption_save,
    }
    img_num = len(set(img_path_save))
    img_id_num = len(set(img_id_save))
    caption_id_num = len(set(caption_id_save))
    caption_num = len(caption_save)

    st = '%s_img_num: %d, %s_img_id_num: %d, %s_caption_id_num: %d, %s_caption_num: %d \n' % (
        dt_type, img_num, dt_type, img_id_num, dt_type, caption_id_num, dt_type, caption_num)
    write_txt(st, os.path.join(output_dir, 'data_message'))

    write_json(data_save, os.path.join(output_dir, '{}_save'.format(dt_type)))
    return data_save


class Occlusion_Adding(object):
    def __init__(self, occlusion_img_dir):
        super(Occlusion_Adding, self).__init__()
        self.occlusion_img_dir = occlusion_img_dir
        self.occlusion_list = ['bench', 'bike', 'car', 'card', 'chair', 'firehydrant', 
                              'motorbike', 'pedestrian', 'post', 'roadsign', 'stone', 
                              'bag', 'kite', 'suitcase', 'umbrella']
        self.area_ratio_dict = {
            'bench': 0.4, 'bike': 0.6, 'car': 0.45, 'card': 0.2, 'chair': 0.65,
            'firehydrant': 0.2, 'motorbike': 0.5, 'pedestrian': 0.5, 'post': 0.5,
            'roadsign': 0.4, 'stone': 0.15, 'bag': 0.2, 'kite': 0.4, 
            'suitcase': 0.1, 'umbrella': 0.25
        }
    
    def __call__(self, holistic_path, save_path):
        occlusion_type = random.choice(self.occlusion_list)
        i = random.randint(1, 4)
        occlusion_path = os.path.join(self.occlusion_img_dir, occlusion_type + str(i) + '.png')
        
        if not os.path.exists(occlusion_path):
            print(f"Warning: Occlusion image not found: {occlusion_path}")
            return None
        
        verse = Image.open(occlusion_path)
        verse = verse.convert("RGBA")
        w_h_ratio = verse.size[0] / verse.size[1]
        holistic_img = Image.open(holistic_path)
        holistic_width = holistic_img.size[0]
        holistic_height = holistic_img.size[1]
        holistic_area = holistic_width * holistic_height
        area_ratio = self.area_ratio_dict[occlusion_type]
        
        new_occlude_area = area_ratio * holistic_area
        occlude_h = int(round(math.sqrt(new_occlude_area / w_h_ratio)))
        occlude_w = int(round(math.sqrt(new_occlude_area * w_h_ratio)))
        verse2 = verse.resize((occlude_w, occlude_h))
        
        if occlusion_type in ['stone', 'motorbike', 'bench', 'bike', 'car', 'card', 'chair', 'post']:
            location = (0, holistic_height - occlude_h)
        elif occlusion_type in ['roadsign', 'firehydrant', 'pedestrian']:
            location = (holistic_width - occlude_w, holistic_height - occlude_h)
        elif occlusion_type in ['umbrella', 'kite']:
            location = (0, 0)
        elif occlusion_type in ['bag']:
            location = (0, int(holistic_height / 2) - occlude_h)
        elif occlusion_type in ['suitcase']:
            location = (holistic_width - occlude_w, int(holistic_height / 2) - occlude_h)
        
        holistic_img.paste(verse2, location, verse2)
        holistic_img.save(save_path)
        return holistic_img


def generate_occlusion(args):
    reid_raw_data = read_json(args.json_root)
    Occlusion = Occlusion_Adding(args.occlusion_path)
    name = args.data_name

    if name == 'ICFG-PEDES':
        for data in reid_raw_data:
            holistic_path = os.path.join(args.data_root, 'imgs', data['file_path'])

            if "test" in holistic_path:
                save_path = holistic_path.replace('test', 'test_occlusion_new')
            elif "train" in holistic_path:
                save_path = holistic_path.replace('train', 'train_occlusion_new')

            save_path2 = os.path.dirname(os.path.abspath(save_path))
            os.makedirs(save_path2, exist_ok=True)

            Occlusion(holistic_path, save_path)

    elif name == 'CUHK-PEDES':
        for data in reid_raw_data:
            holistic_path = os.path.join(args.data_root, 'imgs', data['file_path'])

            if "cam_a" in holistic_path:
                save_path = holistic_path.replace('cam_a', 'cam_a_occlusion_new')
            elif "cam_b" in holistic_path:
                save_path = holistic_path.replace('cam_b', 'cam_b_occlusion_new')
            elif "CUHK01" in holistic_path:
                save_path = holistic_path.replace('CUHK01', 'CUHK01_occlusion_new')
            elif "CUHK03" in holistic_path:
                save_path = holistic_path.replace('CUHK03', 'CUHK03_occlusion_new')
            elif "Market" in holistic_path:
                save_path = holistic_path.replace('Market', 'Market_occlusion_new')
            elif "test_query" in holistic_path:
                save_path = holistic_path.replace('test_query', 'test_query_occlusion_new')
            elif "train_query" in holistic_path:
                save_path = holistic_path.replace('train_query', 'train_query_occlusion_new')

            save_path2 = os.path.dirname(os.path.abspath(save_path))
            os.makedirs(save_path2, exist_ok=True)

            Occlusion(holistic_path, save_path)

    elif name == 'RSTPReid':
        for data in reid_raw_data:
            holistic_path = os.path.join(args.data_root, 'imgs', data['img_path'])

            if "imgs" in holistic_path:
                save_path = holistic_path.replace('imgs', 'imgs_occlusion_new')

            save_path2 = os.path.dirname(os.path.abspath(save_path))
            os.makedirs(save_path2, exist_ok=True)

            Occlusion(holistic_path, save_path)

    return 0


def parse_args():
    parser = argparse.ArgumentParser(description='Generate occluded images for CUHK-PEDES dataset')
    parser.add_argument('--data_name', default='CUHK-PEDES', type=str)
    parser.add_argument('--data_root', 
                        default='请输入你的服务器路径/datasets/CUHK-PEDES',
                        type=str,
                        help='Path to dataset root directory')
    parser.add_argument('--json_root', 
                        default='请输入你的服务器路径/datasets/CUHK-PEDES/reid_raw.json',
                        type=str,
                        help='Path to reid_raw.json file')
    parser.add_argument('--occlusion_path', 
                        default='请输入你的服务器路径/occlusion_img',
                        type=str,
                        help='Path to occlusion image library directory')
    parser.add_argument('--out_root', 
                        default='请输入你的服务器路径/datasets/CUHK-PEDES/processed_data',
                        type=str,
                        help='Path to save processed data')
    parser.add_argument('--min_word_count', default=2, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    print("=" * 60)
    print("Occlusion Image Generation for Dataset")
    print("=" * 60)
    print(f"Data name: {args.data_name}")
    print(f"Data root: {args.data_root}")
    print(f"JSON root: {args.json_root}")
    print(f"Occlusion images: {args.occlusion_path}")
    print(f"Output root: {args.out_root}")
    print("=" * 60)
    
    if not os.path.exists(args.data_root):
        print(f"Error: Data root directory not found: {args.data_root}")
        exit(1)
    
    if not os.path.exists(args.json_root):
        print(f"Error: JSON file not found: {args.json_root}")
        exit(1)
    
    if not os.path.exists(args.occlusion_path):
        print(f"Error: Occlusion image directory not found: {args.occlusion_path}")
        exit(1)
    
    if not os.path.exists(args.out_root):
        os.makedirs(args.out_root)

    print("\nStep 1: Generating occlusion images...")
    generate_occlusion(args)
    print("Step 1: Completed!")

    print("\nStep 2: Splitting dataset...")
    train_json, test_json, val_json = split_json(args)
    print("Step 2: Completed!")
    
    print("\nStep 3: Writing JSON files...")
    write_json(train_json, os.path.join(args.out_root, 'train_json'))
    write_json(test_json, os.path.join(args.out_root, 'test_json'))
    write_json(val_json, os.path.join(args.out_root, 'val_json'))
    print("Step 3: Completed!")
    
    print("\nStep 4: Generating captions...")
    generate_caption(train_json, args, args.out_root, "train")
    generate_caption(test_json, args, args.out_root, "test")
    generate_caption(val_json, args, args.out_root, "val")
    print("Step 4: Completed!")
    
    print("=" * 60)
    print("All steps completed successfully!")
    print("=" * 60)
