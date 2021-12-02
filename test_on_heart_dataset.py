import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader,Subset
import sys
from Dataset import DatasetMaker,get_class_i
from extras import update_lr,disable_dropout
from Medical_predictor_model import ResNet,ResidualBlock
from Medical_Actor import Actor
from Medical_Critic import Critic
from PPO_interface import PPOInterface
from GA_interface import GAInterface
import os
# from tensorboard_maker import make_tensorboard
import random

import numpy as np
import time
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from matplotlib import pyplot
import matplotlib.pyplot as plt

from data_loader import get_iid_loader, get_ood_loader

config = {}
config['outlier_exposure'] = False #True # true/false for adding nonstandard cardiac views during training
config['dataset_name'] = 'heart'

config['sub_iid'] = -1
config['sub_test']= -1

random.seed(10)

device = torch.device("cuda:1")

num_epochs = 1817
learning_rate = 0.001
save_interval = 10
load_predictor_model = True


""" grab dataloader for heart dataset """
mean = [0.122, 0.122, 0.122] # mean and std is pre-computed using training set
std = [0.184, 0.184, 0.184]
train_set, valid_set, train_loader, valid_loader, view_c, view_w = get_iid_loader(
    config['dataset_name'], config['sub_iid'], config['sub_test'], config['outlier_exposure'])
test_set, test_loader = get_ood_loader( (config['dataset_name']+'_test'), config['sub_test'], mean, std)

dataset_sizes = {'train': len(train_set), 'val': len(valid_set), 'test': len(test_set)}
print(dataset_sizes)
dataloaders = {'train':train_loader, 'val':valid_loader, 'test':test_loader}


test_set.add_noise(np.arange(len(test_set)))
# print("test_set.random_noisy_labels_indices",test_set.random_noisy_labels_indices,len(test_set.random_noisy_labels_indices),test_set.vary_labels)


model = ResNet(ResidualBlock, [2, 2, 2],num_classes=2).to(device)

# model.load_state_dict(torch.load("medical_normal_loss/predictor_resnet18.ckpt"))
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_train = 0

actor = Actor(2)
critic = Critic()

interface=PPOInterface(train_set, valid_set,actor,critic,model,device,"medical_new_setting_PPO_logs",
                      load_models=True,
                       controller_save_path='medical_new_setting_PPO_logs/models/PPO_rl_0_5.pth',
                       task_predictor_save_path='medical_new_setting_PPO_logs/checkpoints/predictor_resnet_18_4.ckpt')


# interface=GAInterface(cat_dog_trainset, cat_dog_testset,10,0.2,model,device,"new_setting_GA_logs",
#                       load_models=True, controller_save_path='new_setting_GA_logs/models/GA_rl_0_0.pth',
#                        task_predictor_save_path='new_setting_GA_logs/checkpoints/predictor_resnet_18_0.ckpt')

# PPOInterface.ppo_agent.load()

controller_selection,probs = interface.get_controller_preds_on_holdout(dataloaders['test']) 
noises = []

# print(sum(controller_selection)/len(controller_selection))
interface.task_predictor.eval()
# model.eval()
saved = 1
with torch.no_grad():
    correct = []
    correct_selection = []
    all_labels = []
    # total = 0
    for images, labels,if_noisy in dataloaders['test']:
        images = images.to(device)
        labels = labels.to(device)
        outputs = interface.task_predictor(images)
        all_labels += labels.data.cpu()
        noises += list(if_noisy.cpu().numpy())
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).cpu()
        if saved:
            for n in range(len(list(if_noisy.cpu().numpy()))):
                if if_noisy[n]:
                    plt.imshow(np.transpose(images[n].cpu().numpy(), (1, 2, 0)))
                    plt.savefig("check")
                    saved = 0
            

val_metric = np.mean(np.multiply(np.array(correct),1))
print(noises)
# print(len(all_labels))
selected = 0
t_selected = 0
rejected = 0
t_rejected = 0
for i in range(len(controller_selection)):
    if controller_selection[i] == 1:
        selected += 1
        if noises[i] == 0:
            t_selected += 1
    else:
        rejected += 1
        if noises[i] == 1:
            t_rejected += 1
            
selected_from_each = np.zeros(3)          
print("selection", t_selected/selected, t_rejected/rejected)
selected_correct = []
chosen_quality = np.zeros(4)
none_chosen_quality = np.zeros(4)
chosen_prob = []
none_chosen_prob = []
for i in range(len(controller_selection)):
    if controller_selection[i] == 1:
        selected_correct.append(correct[i])
        selected_from_each[all_labels[i]] += 1
        chosen_quality[noises[i]-1] += 1
        chosen_prob.append(probs[i])
    else:
        none_chosen_quality[noises[i]-1] += 1
        none_chosen_prob.append(probs[i])
print(chosen_prob)
print(chosen_prob)

val_metric = np.mean(np.multiply(np.array(correct),1))
sel_val_metric = np.mean(np.multiply(np.array(selected_correct),1))

# print(selected_from_each)
print(chosen_quality)
print(none_chosen_quality)
print(val_metric,sel_val_metric)

# bins = np.linspace(np.min(probs), np.max(probs), 100)
# pyplot.hist(chosen_prob, bins, alpha=0.5, label='selected_probs')
# pyplot.hist(none_chosen_prob, bins, alpha=0.5, label='selected_probs')
# pyplot.savefig('saved_hist_medical')
# pyplot.clf()











# # print(p.get_controller_preds_on_holdout())


# tensorboard.writer.add_scalar('Loss/Training', train_loss.item(), epoch)
# tensorboard.writer.add_scalar('Accuracy/Training', train_accuracy, epoch)


# model.load_state_dict(torch.load('./normal_loss/predictor_resnet18.ckpt'))
# model.eval()

# with torch.no_grad():
#     correct = []
#     correct_selection = []
#     # total = 0
#     for images, labels, if_noisy in holdout_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         correct += (predicted == labels).cpu()

        
# selected_correct = []
# for i in range(len(controller_selection)):
#     if controller_selection[i] == 1:
#         selected_correct.append(correct[i])

# val_metric = np.mean(np.multiply(np.array(correct),1))
# sel_val_metric = np.mean(np.multiply(np.array(selected_correct),1))

# print(val_metric)


