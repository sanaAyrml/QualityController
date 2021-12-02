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




model = ResNet(ResidualBlock, [2, 2, 2],num_classes=2).to(device)

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_train = 0


if not load_predictor_model:
    
    
    #### log files for multiple runs are NOT overwritten


    log_dir = 'medical_normal_loss' + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)
    # self.model = w #TODO

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run 
    log_f_name_train = log_dir + '/Train_normal_' + "_log_" + str(run_num) + ".csv"

    log_f_name_val = log_dir + '/Valid_normal_' +  "_log_" + str(run_num) + ".csv"



    # logging file
    log_f_train = open(log_f_name_train,"w+")
    log_f_train.write('num_trained_data,epoch,loss,acc\n')

    log_f_valid = open(log_f_name_val,"w+")
    log_f_valid.write('num_trained_data,epoch,acc\n')

    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    best_performance = 0
    
    number_of_total_images = 0
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for i, (images, labels, if_noisy) in enumerate(dataloaders['train']):
            model.train()
            disable_dropout(model)
            images = images.to(device)
            labels = labels.to(device)
            number_of_total_images += len(labels)
            # Forward pass
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            # loss.requres_grad = True

            # Backward and optimizea
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1)  % save_interval == 0:
            print('Accuracy of the model on the train images: {} %'.format(100 * correct / total))
            print ("Epoch [{}/{}], with number of seen data {} Loss: {:.4f}"
                   .format(epoch+1, num_epochs, number_of_total_images, loss.item()))

        if (epoch+1) % 10 == 0:
            log_f_train.write('{},{},{},{}\n'.format(number_of_total_images,epoch,loss.item(), correct/total))
            log_f_train.flush() 
                
        # Decay learning rate
        if (epoch+1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

        if (epoch+1) % save_interval ==  0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels,if_noisy in dataloaders['val']:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print("learning rate is",optimizer.param_groups[0]['lr'])

                print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

            if correct / total > best_performance:
                best_performance = correct / total
                # Save the model checkpoint
                torch.save(model.state_dict(), log_dir + 'predictor_resnet18.ckpt')
            log_f_valid.write('{},{},{}\n'.format(number_of_total_images,epoch, correct / total))
            log_f_valid.flush()
else:
    # model.load_state_dict(torch.load('predictor_resnet18.ckpt'))
    pass
    

actor = Actor(2)
critic = Critic()
# tensorboard = make_tensorboard()
# interface=GAInterface(cat_dog_trainset, cat_dog_testset,10,0.2,model,device,"new_setting_GA_logs")
interface=PPOInterface(train_set,valid_set ,actor,critic,model,device,"medical_new_setting_PPO_logs")
interface.train(1000)

# # print(p.get_controller_preds_on_holdout())


# tensorboard.writer.add_scalar('Loss/Training', train_loss.item(), epoch)
# tensorboard.writer.add_scalar('Accuracy/Training', train_accuracy, epoch)
        