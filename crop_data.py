# -*- encoding: utf-8 -*-
'''
@File    :   crop_data.py
@Time    :   2024
@Description:   Generate cropped images for dataset with full processing pipeline
'''
import numpy as np
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


class ImageCropper(object):
    """
    图像裁剪器
    按位置先验性在上部或下部裁剪图像，裁剪比例25%~40%
    """
    
    def __init__(self, crop_prob=0.3, min_ratio=0.25, max_ratio=0.40):
        """
        初始化裁剪器
        
        Args:
            crop_prob: 裁剪概率 (0-1)，默认0.3表示30%的图像会被裁剪
            min_ratio: 最小裁剪比例，默认0.25 (25%)
            max_ratio: 最大裁剪比例，默认0.40 (40%)
        """
        super(ImageCropper, self).__init__()
        self.crop_prob = crop_prob
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        
        print(f"裁剪器配置:")
        print(f"  - 裁剪概率: {crop_prob * 100:.0f}%")
        print(f"  - 裁剪比例范围: {min_ratio * 100:.0f}% ~ {max_ratio * 100:.0f}%")
        print(f"  - 裁剪位置: 上部 或 下部")
        print(f"  - 填充方式: 黑色边缘填充，居中放置")
    
    def __call__(self, holistic_path, save_path):
        """
        对单张图像进行裁剪和填充
        
        Args:
            holistic_path: 输入图像路径
            save_path: 输出图像保存路径
            
        Returns:
            cropped_img: 裁剪填充后的图像
        """
        try:
            holistic_img = Image.open(holistic_path)
            holistic_width = holistic_img.size[0]
            holistic_height = holistic_img.size[1]
            
            # 按概率决定是否裁剪
            if random.random() >= self.crop_prob:
                # 不裁剪，直接保存原图
                holistic_img.save(save_path)
                return holistic_img
            
            # 随机决定裁剪位置（上部或下部）
            crop_position = random.choice(['top', 'bottom'])
            
            # 随机决定裁剪比例 (25% ~ 40%)
            crop_ratio = random.uniform(self.min_ratio, self.max_ratio)
            
            # 计算裁剪高度
            crop_h = int(holistic_height * crop_ratio)
            
            # 确定裁剪区域
            if crop_position == 'top':
                # 裁剪上部：保留下半部分，放在底部
                y1 = crop_h
                y2 = holistic_height
                crop_description = f"裁剪上部 {crop_h}px (保留下半部分)"
            else:
                # 裁剪下部：保留上半部分，放在顶部
                y1 = 0
                y2 = holistic_height - crop_h
                crop_description = f"裁剪下部 {crop_h}px (保留上半部分)"
            
            # 执行裁剪
            cropped_img = holistic_img.crop((0, y1, holistic_width, y2))
            
            # 创建与原图同尺寸的黑色画布
            padded_img = Image.new('RGB', (holistic_width, holistic_height), (0, 0, 0))
            
            # 将裁剪后的图像保持原位置放置（不居中）
            if crop_position == 'top':
                # 裁剪上部：保留下半部分，放在底部
                pad_y = holistic_height - cropped_img.size[1]
                padded_img.paste(cropped_img, (0, pad_y))
            else:
                # 裁剪下部：保留上半部分，放在顶部
                pad_y = 0
                padded_img.paste(cropped_img, (0, pad_y))
            
            # 保存图像
            padded_img.save(save_path)
            
            return padded_img
            
        except Exception as e:
            print(f"Error processing {holistic_path}: {e}")
            return None


def generate_crop(args):
    """
    Step 1: 生成裁剪图像
    参考process_data.py中generate_occlusion函数的结构
    """
    reid_raw_data = read_json(args.json_root)
    Cropper = ImageCropper(
        crop_prob=args.crop_prob,
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio
    )
    name = args.data_name

    print(f"\n开始生成裁剪图像，数据集: {name}")

    if name == 'ICFG-PEDES':
        for data in reid_raw_data:
            holistic_path = os.path.join(args.data_root, 'imgs', data['file_path'])

            if "test" in holistic_path:
                save_path = holistic_path.replace('test', 'test_crop_new')
            elif "train" in holistic_path:
                save_path = holistic_path.replace('train', 'train_crop_new')

            save_path2 = os.path.dirname(os.path.abspath(save_path))
            os.makedirs(save_path2, exist_ok=True)

            Cropper(holistic_path, save_path)

    elif name == 'CUHK-PEDES':
        for data in reid_raw_data:
            holistic_path = os.path.join(args.data_root, 'imgs', data['file_path'])

            if "cam_a" in holistic_path:
                save_path = holistic_path.replace('cam_a', 'cam_a_crop_new')
            elif "cam_b" in holistic_path:
                save_path = holistic_path.replace('cam_b', 'cam_b_crop_new')
            elif "CUHK01" in holistic_path:
                save_path = holistic_path.replace('CUHK01', 'CUHK01_crop_new')
            elif "CUHK03" in holistic_path:
                save_path = holistic_path.replace('CUHK03', 'CUHK03_crop_new')
            elif "Market" in holistic_path:
                save_path = holistic_path.replace('Market', 'Market_crop_new')
            elif "test_query" in holistic_path:
                save_path = holistic_path.replace('test_query', 'test_query_crop_new')
            elif "train_query" in holistic_path:
                save_path = holistic_path.replace('train_query', 'train_query_crop_new')

            save_path2 = os.path.dirname(os.path.abspath(save_path))
            os.makedirs(save_path2, exist_ok=True)

            Cropper(holistic_path, save_path)

    elif name == 'RSTPReid':
        for data in reid_raw_data:
            holistic_path = os.path.join(args.data_root, 'imgs', data['img_path'])

            if "imgs" in holistic_path:
                save_path = holistic_path.replace('imgs', 'imgs_crop_new')

            save_path2 = os.path.dirname(os.path.abspath(save_path))
            os.makedirs(save_path2, exist_ok=True)

            Cropper(holistic_path, save_path)

    print(f"裁剪图像生成完成！")
    return 0


def split_json(args):
    """
    Step 2: Split dataset into train, test, and val sets
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
    """
    Step 4: Generate caption data
    """
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


def parse_args():
    parser = argparse.ArgumentParser(description='Generate cropped images for dataset with full pipeline')
    parser.add_argument('--data_name', default='CUHK-PEDES', type=str)
    parser.add_argument('--data_root', 
                        default='请输入你的服务器路径/datasets/CUHK-PEDES',
                        type=str,
                        help='Path to dataset root directory')
    parser.add_argument('--json_root', 
                        default='请输入你的服务器路径/datasets/CUHK-PEDES/reid_raw.json',
                        type=str,
                        help='Path to reid_raw.json file')
    parser.add_argument('--out_root', 
                        default='请输入你的服务器路径/datasets/CUHK-PEDES/processed_data',
                        type=str,
                        help='Path to save processed data')
    parser.add_argument('--crop_prob', type=float, default=0.3,
                        help='Crop probability (default: 0.3)')
    parser.add_argument('--min_ratio', type=float, default=0.25,
                        help='Minimum crop ratio (default: 0.25)')
    parser.add_argument('--max_ratio', type=float, default=0.40,
                        help='Maximum crop ratio (default: 0.40)')
    parser.add_argument('--min_word_count', default=2, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    print("=" * 60)
    print("Cropped Image Generation for Dataset")
    print("=" * 60)
    print(f"Data name: {args.data_name}")
    print(f"Data root: {args.data_root}")
    print(f"JSON root: {args.json_root}")
    print(f"Output root: {args.out_root}")
    print(f"Crop probability: {args.crop_prob}")
    print(f"Crop ratio: {args.min_ratio} ~ {args.max_ratio}")
    print("=" * 60)
    
    if not os.path.exists(args.data_root):
        print(f"Error: Data root directory not found: {args.data_root}")
        exit(1)
    
    if not os.path.exists(args.json_root):
        print(f"Error: JSON file not found: {args.json_root}")
        exit(1)
    
    if not os.path.exists(args.out_root):
        os.makedirs(args.out_root)

    print("\nStep 1: Generating cropped images...")
    generate_crop(args)
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
