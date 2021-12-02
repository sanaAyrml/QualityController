# Helper functions to load data for in-distribution and out-of-distribution datasets
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

import heart_loader as heart

def get_iid_loader(database, subset_size_train=-1, subset_size_valid=-1, outlier_exposure=True, seed=2020):
    if database == 'cifar10':
        # mean and standard deviation of channels of CIFAR-10 images
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        train_set, train_loader, _ = _cifar10_loader(mean, std, subset_size_train)
        valid_set, valid_loader, classes = _cifar10_loader(mean, std, subset_size_valid, mode='valid')
        class_weights = np.ones(10)
    if database == 'heart':
        mean = [0.122, 0.122, 0.122]
        std = [0.184, 0.184, 0.184]
        train_set, train_loader, classes, class_weights, qual_c, qual_w = _heart_loader(mean, std, subset_size_train, 'train', outlier_exposure)
        valid_set, valid_loader, _, _, _, _ = _heart_loader(mean, std, subset_size_valid, 'valid', outlier_exposure)
    # if database == 'lung':
        # mean = [0.185, 0.185, 0.185]
        # std = [0.196, 0.196, 0.196]
        # out_set, out_loader = _image_folder_loader("../data/lung_image_dataset", mean, std, subset_size_train)
    if database == 'ob':
        mean = [0.136, 0.136, 0.136]
        std = [0.193, 0.193, 0.193]
        train_set, train_loader, classes, class_weights = _ob_loader(mean, std, subset_size_train, 'train')
        valid_set, valid_loader, _, _ = _ob_loader(mean, std, subset_size_valid, 'valid')
        #class_weights = np.ones(5)
    return train_set, valid_set, train_loader, valid_loader, classes, class_weights#, qual_c, qual_w
    
def get_ood_loader(database, subset_size, target_mean, target_std, seed=2020):
    # extracts images to be trained alongside the training set with the IID data
    # if we'd like to use this for training/test of outlier detection, need to postprocess the batch of labels
    # for training refinement, requires mean/std of IID dataset for whitening
    if database == 'cifar':
        out_set, out_loader, _ = _cifar10_loader(target_mean, target_std, subset_size)
    if database == 'svhn':
        out_set, out_loader = _svhn_ood_loader(target_mean, target_std, subset_size)
    if database == 'dtd':
        out_set, out_loader = _image_folder_loader("../data/dtd/images", target_mean, target_std, subset_size)
        #_dtd_loader(target_mean, target_std, subset_size)
    if database == 'imagenet200':
        out_set, out_loader = _image_folder_loader("../data/tiny-imagenet-200/train", target_mean, target_std, subset_size)
        #_imagenet200_loader(target_mean, target_std, subset_size)
    if database == 'heart':
        out_set, out_loader, _, _, _, _ = _heart_loader(target_mean, target_std, subset_size, 'train')
    if database == 'heart_valid':
        out_set, out_loader, _, _, _, _ = _heart_loader(target_mean, target_std, subset_size, 'valid')
    if database == 'heart_test':
        out_set, out_loader, _, _, _, _ = _heart_loader(target_mean, target_std, subset_size, 'test')
    if database == 'lung':
        out_set, out_loader = _image_folder_loader("../data/lung_image_dataset", target_mean, target_std, subset_size)
        #_lung_loader(target_mean, target_std, subset_size)
    if database == 'ob_test':
        out_set, out_loader, _, _ = _ob_loader(target_mean, target_std, subset_size, 'test')
    return out_set, out_loader

def _cifar10_loader(mean, std, subset_size, mode='train'):
    # if mode == 'train':
        # transform = transforms.Compose(
            # [transforms.Resize(256),
             # transforms.RandomCrop(224),
             # transforms.Grayscale(num_output_channels=3),
             # transforms.ToTensor(),
             # transforms.Normalize(mean, std)])
        # dataset = dset.CIFAR10(root='../data/cifarpy', train=True,
                                            # download=False, transform=transform)
    # else:
        # transform = transforms.Compose(
            # [transforms.Resize(256),
             # transforms.CenterCrop(224),
             # transforms.Grayscale(num_output_channels=3),
             # transforms.ToTensor(),
             # transforms.Normalize(mean, std)])
        # dataset = dset.CIFAR10(root='../data/cifarpy', train=False,
                                            # download=False, transform=transform)
                                            
    # if mode == 'train':
        # transform = _get_transform('train')
        # dataset = dset.CIFAR10(root='../data/cifarpy', train=True,
                                            # download=False, transform=transform)
    # else:
        # transform = _get_transform('valid')
        # dataset = dset.CIFAR10(root='../data/cifarpy', train=False,
                                            # download=False, transform=transform)
    transform = _get_transform(mean, std, mode)
    dataset = dset.CIFAR10(root='../data/cifarpy', train=(mode=='train'),
                                        download=False, transform=transform)
    dataset_1 = _take_subset(dataset, subset_size)
    data_loader = torch.utils.data.DataLoader(dataset_1, batch_size=4,
                                              shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = 10
    return dataset_1, data_loader, classes
    
def _ob_loader(mean, std, subset_size, mode='train', outlier_exposure=True):
    transform = _get_transform(mean, std, mode, to_pil_image=True)
    dataset = ob.OB("../data/ob_image_dataset", transform, mode)
    classes = dataset.get_classes()
    class_weights = dataset.get_class_weights()
    
    dataset_1 = _take_subset(dataset, subset_size)
    # data_loader = torch.utils.data.DataLoader(dataset_1, batch_size=4,
                                              # shuffle=True, num_workers=2)
    data_loader = torch.utils.data.DataLoader(dataset_1, batch_size=4,
                                              shuffle=True, num_workers=2)
    
    return dataset, data_loader, classes, class_weights
    
def _heart_loader(mean, std, subset_size, mode='train', outlier_exposure=False):
    transform = _get_transform(mean, std, mode, to_pil_image=True)
    if mode == 'train':
        dataset = heart.Heart('../database_path/train_labels_2.txt', transform, outlier_exposure)
    elif mode == 'valid':
        dataset = heart.Heart('../database_path/valid_labels_2.txt', transform, outlier_exposure)
    elif mode == 'test':
        dataset = heart.Heart('../database_path/test_labels_2.txt', transform, outlier_exposure)
    view_c, qual_c = dataset.get_classes()
    view_w, qual_w = dataset.get_label_sampling_weights()
    view_w_per_item = dataset.get_label_sampling_weights_per_item()
    
    #dataset_1 = _take_subset(dataset, subset_size)
    # data_loader = torch.utils.data.DataLoader(dataset_1, batch_size=4,
                                              # shuffle=True, num_workers=2)
    bs = 4
    if (subset_size == -1):
        if mode=='train':
            samp = torch.utils.data.WeightedRandomSampler(view_w_per_item, num_samples=len(dataset) )
        else:
            samp = torch.utils.data.SequentialSampler(dataset)
    else:
        if mode=='train':
            samp = torch.utils.data.WeightedRandomSampler(view_w_per_item, num_samples=subset_size)
        else:
            samp = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=subset_size)
        
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, sampler=samp, num_workers=2)
    
    return dataset, data_loader, view_c, view_w, qual_c, qual_w

def _svhn_ood_loader(mean, std, subset_size):
    transform = _get_transform(mean, std, 'valid')
    dataset = svhn.SVHN(root='../data/svhn/', split="test", transform=transform, download=False)
    dataset_1 = _take_subset(dataset, subset_size)
    data_loader = torch.utils.data.DataLoader(dataset_1, batch_size=4,
                                              shuffle=True, num_workers=2)
    return dataset_1, data_loader
    
def _image_folder_loader(folder_name, mean, std, subset_size):
    transform = _get_transform(mean, std, 'valid')
    dataset = dset.ImageFolder(root=folder_name,
                            transform=transform)
    dataset_1 = _take_subset(dataset, subset_size)
    data_loader = torch.utils.data.DataLoader(dataset_1, batch_size=4, shuffle=True,
                                             num_workers=2)
    return dataset_1, data_loader

# def _dtd_loader(mean, std, subset_size):
    # transform = transforms.Compose(
            # [transforms.Resize((224, 224)), 
             # transforms.Grayscale(num_output_channels=3),
             # transforms.ToTensor(), 
             # transforms.Normalize(mean, std)])
    # dataset = dset.ImageFolder(root="../data/dtd/images",
                            # transform=transform)
    # dataset_1 = _take_subset(dataset, subset_size)
    # data_loader = torch.utils.data.DataLoader(dataset_1, batch_size=4, shuffle=True,
                                             # num_workers=2)
    # return dataset_1, data_loader

# def _imagenet200_loader(mean, std, subset_size):
    # transform = transforms.Compose(
            # [transforms.Resize(224), 
             # transforms.Grayscale(num_output_channels=3),
             # transforms.ToTensor(), 
             # transforms.Normalize(mean, std)])
    # dataset = dset.ImageFolder(root="../data/tiny-imagenet-200/train",transform=transform)
    # dataset_1 = _take_subset(dataset, subset_size)
    # data_loader = torch.utils.data.DataLoader(dataset_1, batch_size=4, shuffle=True,
                                             # num_workers=2)
    # return dataset_1, data_loader
    
# def _lung_loader(mean, std, subset_size):
    # transform = transforms.Compose(
            # [transforms.Resize((224,224)), 
             # transforms.Grayscale(num_output_channels=3),
             # transforms.ToTensor(), 
             # transforms.Normalize(mean, std)])
    # dataset = dset.ImageFolder(root="../data/lung_image_dataset",transform=transform)
    # dataset_1 = _take_subset(dataset, subset_size)
    # data_loader = torch.utils.data.DataLoader(dataset_1, batch_size=4, shuffle=True,
                                             # num_workers=2)
    # return dataset_1, data_loader

def _get_transform(mean, std, mode='valid', to_pil_image=False):
    # if mode=='train':
        # t_transform = transforms.Compose(
            # [transforms.Resize((256,256)),
             # transforms.RandomRotation(5),
             # transforms.RandomCrop((224,224)),
             # transforms.Grayscale(num_output_channels=3),
             # transforms.ToTensor()])#,
             # #transforms.Normalize(mean, std)])
    # else: # mode=='valid'
        # t_transform = transforms.Compose(
            # [transforms.Resize((224,224)),
             # transforms.Grayscale(num_output_channels=3),
             # transforms.ToTensor()])#, 
             # #transforms.Normalize(mean, std)])
    if mode=='train':
        aug_transform = transforms.Compose(
            [transforms.Resize((256,256)),
             transforms.RandomRotation(5),
             transforms.RandomCrop((224,224)) ])
    else: # mode=='valid'
        aug_transform = transforms.Resize((224,224))
    t_transform = transforms.Compose(
        [aug_transform,
         transforms.Grayscale(num_output_channels=3),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)
        ])
    if to_pil_image:
        t_transform = transforms.Compose([transforms.ToPILImage(), 
                                            t_transform])
    return t_transform

def _take_subset(dataset, subset_size):
    np.random.seed(2020)
    if (subset_size == -1):
        dataset_1 = dataset # use full dataset
    else: # use subset, if subset_size > dataset_size, this will implement oversampling
        dataset_1 = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), subset_size) )
    return dataset_1