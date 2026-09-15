import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from data.utils import pre_caption


class cuhk_pede_train_with_occlusion_and_crop(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt='',
                 occlusion_img_dir=None, crop_img_dir=None,
                 occlusion_prob=0.3, crop_prob=0.3, original_prob=0.4):
        from data.cuhk_dataset import split_CUHK_PEDE
        train_list, _, _ = split_CUHK_PEDE()
        self.annotation = train_list
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.occlusion_prob = occlusion_prob
        self.crop_prob = crop_prob
        self.original_prob = original_prob

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
        image = Image.open(os.path.join(self.image_root, ann['file_path']))

        rand_val = random.random()

        if rand_val < self.occlusion_prob:
            branch = 'occlusion'
        elif rand_val < self.occlusion_prob + self.crop_prob:
            branch = 'crop'
        else:
            branch = 'original'

        if branch == 'occlusion':
            occlusion_path = self._get_occlusion_path(os.path.join(self.image_root, ann['file_path']))
            if os.path.exists(occlusion_path):
                image = Image.open(occlusion_path).convert('RGB')
            else:
                image = Image.open(os.path.join(self.image_root, ann['file_path'])).convert('RGB')
        elif branch == 'crop':
            crop_path = self._get_crop_path(os.path.join(self.image_root, ann['file_path']))
            if os.path.exists(crop_path):
                image = Image.open(crop_path).convert('RGB')
            else:
                image = Image.open(os.path.join(self.image_root, ann['file_path'])).convert('RGB')
        else:
            image = Image.open(os.path.join(self.image_root, ann['file_path'])).convert('RGB')

        image = self.transform(image)
        captions = ann['captions']
        if isinstance(captions, list):
            captions = captions[0] if len(captions) > 0 else ""
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

    def _get_crop_path(self, original_path):
        crop_path = original_path
        if 'cam_a' in crop_path:
            crop_path = crop_path.replace('cam_a', 'cam_a_crop_new')
        elif 'cam_b' in crop_path:
            crop_path = crop_path.replace('cam_b', 'cam_b_crop_new')
        elif 'CUHK01' in crop_path:
            crop_path = crop_path.replace('CUHK01', 'CUHK01_crop_new')
        elif 'CUHK03' in crop_path:
            crop_path = crop_path.replace('CUHK03', 'CUHK03_crop_new')
        elif 'Market' in crop_path:
            crop_path = crop_path.replace('Market', 'Market_crop_new')
        elif 'train_query' in crop_path:
            crop_path = crop_path.replace('train_query', 'train_query_crop_new')
        elif 'test' in crop_path:
            crop_path = crop_path.replace('test', 'test_crop_new')
        elif 'train' in crop_path:
            crop_path = crop_path.replace('train', 'train_crop_new')
        return crop_path


class icfg_pede_train_with_occlusion_and_crop(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt='',
                 occlusion_img_dir=None, crop_img_dir=None,
                 occlusion_prob=0.3, crop_prob=0.3, original_prob=0.4):
        from data.icfg_dataset import split_ICFG_PEDE
        train_list, _, _ = split_ICFG_PEDE()
        self.annotation = train_list
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.occlusion_prob = occlusion_prob
        self.crop_prob = crop_prob
        self.original_prob = original_prob

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
        image = Image.open(os.path.join(self.image_root, ann['file_path']))

        rand_val = random.random()

        if rand_val < self.occlusion_prob:
            branch = 'occlusion'
        elif rand_val < self.occlusion_prob + self.crop_prob:
            branch = 'crop'
        else:
            branch = 'original'

        if branch == 'occlusion':
            occlusion_path = self._get_occlusion_path(os.path.join(self.image_root, ann['file_path']))
            if os.path.exists(occlusion_path):
                image = Image.open(occlusion_path).convert('RGB')
        elif branch == 'crop':
            crop_path = self._get_crop_path(os.path.join(self.image_root, ann['file_path']))
            if os.path.exists(crop_path):
                image = Image.open(crop_path).convert('RGB')

        image = self.transform(image)
        captions = ann['captions']
        if isinstance(captions, list):
            captions = captions[0] if len(captions) > 0 else ""
        captions = self.prompt + pre_caption(captions, self.max_words)
        return image, captions, self.img_ids[ann['id']]

    def _get_occlusion_path(self, original_path):
        occlusion_path = original_path
        if 'test' in occlusion_path:
            occlusion_path = occlusion_path.replace('test', 'test_occlusion_new')
        elif 'train' in occlusion_path:
            occlusion_path = occlusion_path.replace('train', 'train_occlusion_new')
        return occlusion_path

    def _get_crop_path(self, original_path):
        crop_path = original_path
        if 'test' in crop_path:
            crop_path = crop_path.replace('test', 'test_crop_new')
        elif 'train' in crop_path:
            crop_path = crop_path.replace('train', 'train_crop_new')
        return crop_path


class rstp_pede_train_with_occlusion_and_crop(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt='',
                 occlusion_img_dir=None, crop_img_dir=None,
                 occlusion_prob=0.3, crop_prob=0.3, original_prob=0.4):
        from data.rstp_dataset import split_RSTP_PEDE
        train_list, _, _ = split_RSTP_PEDE()
        self.annotation = train_list
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.occlusion_prob = occlusion_prob
        self.crop_prob = crop_prob
        self.original_prob = original_prob

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
        image = Image.open(os.path.join(self.image_root, ann['img_path']))

        rand_val = random.random()

        if rand_val < self.occlusion_prob:
            branch = 'occlusion'
        elif rand_val < self.occlusion_prob + self.crop_prob:
            branch = 'crop'
        else:
            branch = 'original'

        if branch == 'occlusion':
            occlusion_path = self._get_occlusion_path(os.path.join(self.image_root, ann['img_path']))
            if os.path.exists(occlusion_path):
                image = Image.open(occlusion_path).convert('RGB')
        elif branch == 'crop':
            crop_path = self._get_crop_path(os.path.join(self.image_root, ann['img_path']))
            if os.path.exists(crop_path):
                image = Image.open(crop_path).convert('RGB')

        image = self.transform(image)
        captions = ann['captions']
        if isinstance(captions, list):
            captions = captions[0] if len(captions) > 0 else ""
        captions = self.prompt + pre_caption(captions, self.max_words)
        return image, captions, self.img_ids[ann['id']]

    def _get_occlusion_path(self, original_path):
        occlusion_path = original_path
        if 'RSTP' in occlusion_path:
            occlusion_path = occlusion_path.replace('RSTP', 'RSTP_occlusion_new')
        return occlusion_path

    def _get_crop_path(self, original_path):
        crop_path = original_path
        if 'RSTP' in crop_path:
            crop_path = crop_path.replace('RSTP', 'RSTP_crop_new')
        return crop_path


class mix_pede_train_with_occlusion_and_crop(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt='',
                 occlusion_img_dir=None, crop_img_dir=None,
                 occlusion_prob=0.3, crop_prob=0.3, original_prob=0.4):
        from data.cuhk_dataset import split_CUHK_PEDE
        from data.icfg_dataset import split_ICFG_PEDE

        train_list, _, _ = split_CUHK_PEDE()
        train_list2, _, _ = split_ICFG_PEDE()
        self.annotation = train_list + train_list2
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.occlusion_prob = occlusion_prob
        self.crop_prob = crop_prob
        self.original_prob = original_prob

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
        image = Image.open(os.path.join(self.image_root, ann['file_path']))

        rand_val = random.random()

        if rand_val < self.occlusion_prob:
            branch = 'occlusion'
        elif rand_val < self.occlusion_prob + self.crop_prob:
            branch = 'crop'
        else:
            branch = 'original'

        if branch == 'occlusion':
            occlusion_path = self._get_occlusion_path(os.path.join(self.image_root, ann['file_path']))
            if os.path.exists(occlusion_path):
                image = Image.open(occlusion_path).convert('RGB')
            else:
                image = Image.open(os.path.join(self.image_root, ann['file_path'])).convert('RGB')
        elif branch == 'crop':
            crop_path = self._get_crop_path(os.path.join(self.image_root, ann['file_path']))
            if os.path.exists(crop_path):
                image = Image.open(crop_path).convert('RGB')
            else:
                image = Image.open(os.path.join(self.image_root, ann['file_path'])).convert('RGB')
        else:
            image = Image.open(os.path.join(self.image_root, ann['file_path'])).convert('RGB')

        image = self.transform(image)
        captions = ann['captions']
        if isinstance(captions, list):
            captions = captions[0] if len(captions) > 0 else ""
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

    def _get_crop_path(self, original_path):
        crop_path = original_path
        if 'cam_a' in crop_path:
            crop_path = crop_path.replace('cam_a', 'cam_a_crop_new')
        elif 'cam_b' in crop_path:
            crop_path = crop_path.replace('cam_b', 'cam_b_crop_new')
        elif 'CUHK01' in crop_path:
            crop_path = crop_path.replace('CUHK01', 'CUHK01_crop_new')
        elif 'CUHK03' in crop_path:
            crop_path = crop_path.replace('CUHK03', 'CUHK03_crop_new')
        elif 'Market' in crop_path:
            crop_path = crop_path.replace('Market', 'Market_crop_new')
        elif 'train_query' in crop_path:
            crop_path = crop_path.replace('train_query', 'train_query_crop_new')
        elif 'test' in crop_path:
            crop_path = crop_path.replace('test', 'test_crop_new')
        elif 'train' in crop_path:
            crop_path = crop_path.replace('train', 'train_crop_new')
        return crop_path