import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from data.utils import pre_caption


class cuhk_pede_train_with_occlusion(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt='',
                 occlusion_img_dir=None, occlusion_prob=0.5):
        from data.cuhk_dataset import split_CUHK_PEDE
        train_list, _, _ = split_CUHK_PEDE()
        self.annotation = train_list
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.occlusion_prob = occlusion_prob

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['img_path'])

        if random.random() < self.occlusion_prob:
            occlusion_path = self._get_occlusion_path(image_path)
            if os.path.exists(occlusion_path):
                image = Image.open(occlusion_path).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')

        image = self.transform(image)
        captions = ann['captions']
        captions = self.prompt + pre_caption(captions, self.max_words)
        return image, captions, self.img_ids[ann['id']]

    def _get_occlusion_path(self, original_path):
        occlusion_path = original_path
        if 'cam_a' in occlusion_path:
            occlusion_path = occlusion_path.replace('cam_a', 'cam_a_occlusion_new')
        elif 'cam_b' in occlusion_path:
            occlusion_path = occlusion_path.replace('cam_b', 'cam_b_occlusion_new')
        elif 'CUHK01' in occlusion_path:
            occlusion_path = occlusion_path.replace('CUHK01', 'CUHK01_occlusion_new')
        elif 'CUHK03' in occlusion_path:
            occlusion_path = occlusion_path.replace('CUHK03', 'CUHK03_occlusion_new')
        elif 'Market' in occlusion_path:
            occlusion_path = occlusion_path.replace('Market', 'Market_occlusion_new')
        elif 'train_query' in occlusion_path:
            occlusion_path = occlusion_path.replace('train_query', 'train_query_occlusion_new')
        return occlusion_path


class mix_pede_train_with_occlusion(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt='',
                 occlusion_img_dir=None, occlusion_prob=0.5):
        from data.cuhk_dataset import split_CUHK_PEDE
        from data.icfg_dataset import split_ICFG_PEDE

        train_list, _, _ = split_CUHK_PEDE()
        self.annotation = train_list
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.occlusion_prob = occlusion_prob

        self.img_ids = {}
        n = 0
        for i, ann in enumerate(self.annotation):
            img_id = ann['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
            self.annotation[i]['file_path'] = os.path.join(image_root, self.annotation[i]['file_path'])

        train_list, _, _ = split_ICFG_PEDE()
        for i, ann in enumerate(train_list):
            train_list[i]['id'] = train_list[i]['id'] + 20000
            img_id = train_list[i]['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
            train_list[i]['file_path'] = os.path.join(image_root.replace('CUHK-PEDES', 'ICFG-PEDES'), train_list[i]['file_path'])
            train_list[i]['captions'] = ann['captions'][0]
        self.annotation = self.annotation + train_list

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = ann['file_path']

        if random.random() < self.occlusion_prob:
            occlusion_path = self._get_occlusion_path(image_path)
            if os.path.exists(occlusion_path):
                image = Image.open(occlusion_path).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')

        image = self.transform(image)
        captions = ann['captions']
        captions = self.prompt + pre_caption(captions, self.max_words)
        return image, captions, self.img_ids[ann['id']]

    def _get_occlusion_path(self, original_path):
        occlusion_path = original_path
        if 'cam_a' in occlusion_path:
            occlusion_path = occlusion_path.replace('cam_a', 'cam_a_occlusion_new')
        elif 'cam_b' in occlusion_path:
            occlusion_path = occlusion_path.replace('cam_b', 'cam_b_occlusion_new')
        elif 'CUHK01' in occlusion_path:
            occlusion_path = occlusion_path.replace('CUHK01', 'CUHK01_occlusion_new')
        elif 'CUHK03' in occlusion_path:
            occlusion_path = occlusion_path.replace('CUHK03', 'CUHK03_occlusion_new')
        elif 'Market' in occlusion_path:
            occlusion_path = occlusion_path.replace('Market', 'Market_occlusion_new')
        elif 'train_query' in occlusion_path:
            occlusion_path = occlusion_path.replace('train_query', 'train_query_occlusion_new')
        return occlusion_path


class icfg_pede_train_with_occlusion(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt='',
                 occlusion_img_dir=None, occlusion_prob=0.5):
        from data.icfg_dataset import split_ICFG_PEDE
        train_list, _, _ = split_ICFG_PEDE()
        self.annotation = train_list
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.occlusion_prob = occlusion_prob

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, 'imgs', ann['file_path'])

        if random.random() < self.occlusion_prob:
            occlusion_path = self._get_occlusion_path(image_path)
            if os.path.exists(occlusion_path):
                image = Image.open(occlusion_path).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')

        image = self.transform(image)
        captions = ann['captions']
        captions = self.prompt + pre_caption(captions, self.max_words)
        return image, captions, self.img_ids[ann['id']]

    def _get_occlusion_path(self, original_path):
        occlusion_path = original_path
        if 'test' in occlusion_path:
            occlusion_path = occlusion_path.replace('test', 'test_occlusion_new')
        elif 'train' in occlusion_path:
            occlusion_path = occlusion_path.replace('train', 'train_occlusion_new')
        return occlusion_path


class rstp_pede_train_with_occlusion(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt='',
                 occlusion_img_dir=None, occlusion_prob=0.5):
        from data.rstp_dataset import split_RSTPReid
        train_list, _, _ = split_RSTPReid()
        self.annotation = train_list
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.occlusion_prob = occlusion_prob

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, 'imgs', ann['img_path'])

        if random.random() < self.occlusion_prob:
            occlusion_path = self._get_occlusion_path(image_path)
            if os.path.exists(occlusion_path):
                image = Image.open(occlusion_path).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')

        image = self.transform(image)
        captions = ann['captions']
        captions = self.prompt + pre_caption(captions, self.max_words)
        return image, captions, self.img_ids[ann['id']]

    def _get_occlusion_path(self, original_path):
        occlusion_path = original_path
        if 'RSTP' in occlusion_path:
            occlusion_path = occlusion_path.replace('RSTP', 'RSTP_occlusion_new')
        return occlusion_path