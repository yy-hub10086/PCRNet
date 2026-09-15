import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.cuhk_dataset import cuhk_pede_train, cuhk_pede_caption_eval, cuhk_pede_retrieval_eval,cuhk_pede_trainset_eval,mix_pede_train
from data.icfg_dataset import icfg_pede_train, icfg_pede_retrieval_eval
from data.rstp_dataset import rstp_pede_train, rstp_pede_retrieval_eval
from data.occlusion_dataset import cuhk_pede_train_with_occlusion, mix_pede_train_with_occlusion, icfg_pede_train_with_occlusion, rstp_pede_train_with_occlusion
from data.crop_dataset import cuhk_pede_train_with_occlusion_and_crop, mix_pede_train_with_occlusion_and_crop, icfg_pede_train_with_occlusion_and_crop, rstp_pede_train_with_occlusion_and_crop
from transform.randaugment import RandomAugment

def create_dataset(dataset, config, min_scale=0.5):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    if type(config['image_size']) == int:
        image_size = (config['image_size'], config['image_size'])
    elif type(config['image_size']) == list or type(config['image_size']) == tuple:
        image_size = (config['image_size'][0], config['image_size'][1])

    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Pad(10),
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(scale=(0.02, 0.4), value=(0.48145466, 0.4578275, 0.40821073)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    if  dataset=='caption_cuhk':
        train_dataset = cuhk_pede_train(transform_train, config['image_root'])
        val_dataset = cuhk_pede_caption_eval(transform_test, config['image_root'], 'val')
        test_dataset = cuhk_pede_caption_eval(transform_test, config['image_root'], 'test')
        return train_dataset, val_dataset, test_dataset

    elif dataset=='retrieval_cuhk':
        if 'use_occlusion_and_crop' in config and config['use_occlusion_and_crop']:
            occlusion_prob = config.get('occlusion_prob', 0.3)
            crop_prob = config.get('crop_prob', 0.3)
            original_prob = config.get('original_prob', 0.4)
            train_dataset = cuhk_pede_train_with_occlusion_and_crop(transform_train, config['image_root'], 
                                                              occlusion_prob=occlusion_prob,
                                                              crop_prob=crop_prob,
                                                              original_prob=original_prob)
        elif 'use_occlusion' in config and config['use_occlusion']:
            occlusion_prob = config.get('occlusion_prob', 0.5)
            train_dataset = cuhk_pede_train_with_occlusion(transform_train, config['image_root'], 
                                                       occlusion_prob=occlusion_prob)
        else:
            train_dataset = cuhk_pede_train(transform_train, config['image_root'])
        val_dataset = cuhk_pede_retrieval_eval(transform_test, config['image_root'], 'val')
        test_dataset = cuhk_pede_retrieval_eval(transform_test, config['image_root'], 'test')
        return train_dataset, val_dataset, test_dataset

    elif dataset=='cuhk_trainset_eval':
        train_dataset = cuhk_pede_train(transform_train, config['image_root'])
        test_dataset = cuhk_pede_retrieval_eval(transform_test, config['image_root'], 'test')
        val_dataset = cuhk_pede_trainset_eval(transform_test,config['image_root'])
        return train_dataset,val_dataset,test_dataset


    elif dataset=='retrieval_icfg':
        if 'use_occlusion_and_crop' in config and config['use_occlusion_and_crop']:
            occlusion_prob = config.get('occlusion_prob', 0.3)
            crop_prob = config.get('crop_prob', 0.3)
            original_prob = config.get('original_prob', 0.4)
            train_dataset = icfg_pede_train_with_occlusion_and_crop(transform_train, config['image_root'], 
                                                              occlusion_prob=occlusion_prob,
                                                              crop_prob=crop_prob,
                                                              original_prob=original_prob)
        elif 'use_occlusion' in config and config['use_occlusion']:
            occlusion_prob = config.get('occlusion_prob', 0.5)
            train_dataset = icfg_pede_train_with_occlusion(transform_train, config['image_root'], 
                                                       occlusion_prob=occlusion_prob)
        else:
            train_dataset = icfg_pede_train(transform_train, config['image_root'])
        val_dataset = icfg_pede_retrieval_eval(transform_test, config['image_root'], 'val')
        test_dataset = icfg_pede_retrieval_eval(transform_test, config['image_root'], 'test')
        return train_dataset, val_dataset, test_dataset

    elif dataset=='retrieval_rstp':
        if 'use_occlusion_and_crop' in config and config['use_occlusion_and_crop']:
            occlusion_prob = config.get('occlusion_prob', 0.3)
            crop_prob = config.get('crop_prob', 0.3)
            original_prob = config.get('original_prob', 0.4)
            train_dataset = rstp_pede_train_with_occlusion_and_crop(transform_train, config['image_root'], 
                                                              occlusion_prob=occlusion_prob,
                                                              crop_prob=crop_prob,
                                                              original_prob=original_prob)
        elif 'use_occlusion' in config and config['use_occlusion']:
            occlusion_prob = config.get('occlusion_prob', 0.5)
            train_dataset = rstp_pede_train_with_occlusion(transform_train, config['image_root'], 
                                                       occlusion_prob=occlusion_prob)
        else:
            train_dataset = rstp_pede_train(transform_train, config['image_root'])
        val_dataset = rstp_pede_retrieval_eval(transform_test, config['image_root'], 'val')
        test_dataset = rstp_pede_retrieval_eval(transform_test, config['image_root'], 'test')
        return train_dataset, val_dataset, test_dataset
    
    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

