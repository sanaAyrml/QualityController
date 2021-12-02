import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader,Subset
import sys
from Dataset import DatasetMaker,get_class_i
from extras import update_lr,disable_dropout
from predictor_model import ResNet,ResidualBlock
from Actor import Actor
from Critic import Critic
from PPO_interface import PPOInterface
from GA_interface import GAInterface
import os
import random
from matplotlib import pyplot

from Dataset import DatasetMaker,get_class_i
from Dataset_Noisy import NoisyDatasetMaker,get_class_i
# from tensorboard_maker import make_tensorboard

random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 1817
learning_rate = 0.001
save_interval = 10
load_predictor_model = True

transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip()])

transform2 = transforms.Compose([
                                transforms.ToTensor()])


# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='~/continual learning',
                                             train=True,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='~/continual learning',
                                            train=False)

x_train = train_dataset.data

x_test = test_dataset.data
y_train = train_dataset.targets
y_test = test_dataset.targets
classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
             'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

trainset = \
       NoisyDatasetMaker(
        [get_class_i(x_train, y_train, classDict['cat']),
         get_class_i(x_train, y_train, classDict['dog'])],
        transform
    )

cat_dog_holdoutset = \
NoisyDatasetMaker(
        [get_class_i(x_test, y_test, classDict['cat']),
         get_class_i(x_test, y_test, classDict['dog'])],
        transform2
    )


indices = np.load("save_test_noise_indices.npy")
# cat_dog_holdoutset.add_noise(indices,0.5)






holdout_loader = torch.utils.data.DataLoader(dataset=cat_dog_holdoutset,
                                          batch_size=100, 
                                          shuffle=False)


model = ResNet(ResidualBlock, [2, 2, 2],num_classes=2).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_train = 0

# actor = Actor(2)
# critic = Critic()

# interface=PPOInterface(trainset, trainset,actor,critic,model,device,"PPO_logs_noisy_2_class_cv",
#                       load_models=True,
#                        controller_save_path='PPO_logs_noisy_2_class_cv/1/models/PPO_rl_0_0.pth',
#                        task_predictor_save_path='PPO_logs_noisy_2_class_cv/1/checkpoints/predictor_resnet_18_0.ckpt')


interface=GAInterface(trainset, trainset,10,0.2,model,device,"GA_logs_cv",
                      load_models=True, controller_save_path='GA_logs_cv/2/models/GA_rl_0_0.pth',
                       task_predictor_save_path='GA_logs_cv/2/checkpoints/predictor_resnet_18_0.ckpt')



controller_selection = (-1*interface.get_controller_preds_on_holdout(holdout_loader))+1 
# controller_selection,_ = interface.get_controller_preds_on_holdout(holdout_loader) 
# controller_selection = -1*controller_selection + 1


noises = []


print("percentage of data chosen: ",sum(controller_selection)/len(controller_selection))
interface.task_predictor.eval()
with torch.no_grad():
    correct = []
    correct_selection = []
    all_labels = []
    # total = 0
    for images, labels,if_noisy in holdout_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = interface.task_predictor(images)
        all_labels += labels.data.cpu()
        noises += if_noisy
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).cpu()

val_metric = np.mean(np.multiply(np.array(correct),1))
print("performence of the whole dataset: ",val_metric)
selected = 0
t_selected = 0
rejected = 0
t_rejected = 0
selected_probs =[]
none_selected_probs =[]
for i in range(len(controller_selection)):
    if controller_selection[i] >= 0.5:
        selected += 1
        if noises[i] == 0:
            t_selected += 1
    else:
        rejected += 1
        if noises[i] == 1:
            t_rejected += 1

print("selection", t_selected/selected, t_rejected/rejected)
selected_correct = []
for i in range(len(controller_selection)):
    if controller_selection[i] == 1:
        selected_correct.append(correct[i])


sel_val_metric = np.mean(np.multiply(np.array(selected_correct),1))

print("performence of the selected dataset: ",sel_val_metric)


# bins = np.linspace(np.min(selection_prob), np.max(selection_prob), 100)

# pyplot.hist(selected_probs, bins, alpha=0.5, label='selected_probs')
# pyplot.hist(none_selected_probs, bins, alpha=0.5, label='none_selected_probs')
# pyplot.legend(loc='upper right')
# pyplot.savefig('continous_saved_hist_added_class')
# pyplot.clf()







# model.load_state_dict(torch.load('./2_normal_loss_cv_2_class/0/predictor_resnet18_0.ckpt'))
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

        


# val_metric = np.mean(np.multiply(np.array(correct),1))

# print(val_metric)

