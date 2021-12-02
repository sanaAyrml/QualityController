import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader,Subset
import sys
from Dataset_Noisy import NoisyDatasetMaker,get_class_i
from extras import update_lr,disable_dropout
from predictor_model import ResNet,ResidualBlock
from Actor import Actor
from Critic import Critic
from PPO_interface import PPOInterface
from GA_interface import GAInterface
import os
# from tensorboard_maker import make_tensorboard
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

random.seed(10)

device = torch.device("cuda:1")

num_epochs = 150
learning_rate = 0.001
save_interval = 10
load_predictor_model = False
warm_up_noisy = False

transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip()])


# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='~/continual learning',
                                             train=True,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='~/continual learning',
                                            train=False)

x_train = train_dataset.data
print(type(x_train[0]))
print(x_train[0].shape)
x_test = test_dataset.data
y_train = train_dataset.targets
y_test = test_dataset.targets
classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
             'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

# trainset = \
#     NoisyDatasetMaker(
#         [get_class_i(x_train, y_train, classDict['cat']),
#          get_class_i(x_train, y_train, classDict['dog']),
#         get_class_i(x_train, y_train, classDict['plane']),
#         get_class_i(x_train, y_train, classDict['car']),
#         get_class_i(x_train, y_train, classDict['bird']),
#         get_class_i(x_train, y_train, classDict['deer']),
#         get_class_i(x_train, y_train, classDict['frog']),
#         get_class_i(x_train, y_train, classDict['horse']),
#         get_class_i(x_train, y_train, classDict['ship']),
#         get_class_i(x_train, y_train, classDict['truck'])],
#         transform
#     )
# holdoutset = \
#     NoisyDatasetMaker(
#         [get_class_i(x_test, y_test, classDict['cat']),
#          get_class_i(x_test, y_test, classDict['dog']),
#         get_class_i(x_test, y_test, classDict['plane']),
#         get_class_i(x_test, y_test, classDict['car']),
#         get_class_i(x_test, y_test, classDict['bird']),
#         get_class_i(x_test, y_test, classDict['deer']),
#         get_class_i(x_test, y_test, classDict['frog']),
#         get_class_i(x_test, y_test, classDict['horse']),
#         get_class_i(x_test, y_test, classDict['ship']),
#         get_class_i(x_test, y_test, classDict['truck'])],
#         transform
#     )


trainset = \
    NoisyDatasetMaker(
        [get_class_i(x_train, y_train, classDict['cat']),
         get_class_i(x_train, y_train, classDict['dog'])],
        transform
    )
holdoutset = \
    NoisyDatasetMaker(
        [get_class_i(x_test, y_test, classDict['cat']),
         get_class_i(x_test, y_test, classDict['dog'])],
        transform
    )

print(len(trainset),len(holdoutset))

indices = np.arange(len(trainset))
np.random.shuffle(indices)
print(int((4*len(indices))/5))
training_indices, training_indices_2 = indices[:int((4*len(indices))/5)], indices[int((4*len(indices))/5):]
print(len(training_indices),len(training_indices_2))



kf = KFold(n_splits=3, shuffle=True)
print(kf)

train_indices_save = np.load("save_train_indices_2_class.npy",allow_pickle=True)
val_indices_save = np.load("save_val_indices_2_class.npy",allow_pickle=True)
# np.save("save_train_indices_2_2_class.npy",training_indices_2)
# train_indices_save = []
# val_indices_save = []
#     train_indices_save.append(train_indices)
#     val_indices_save.append(val_indices)
# np.save("save_train_indices.npy",np.array(train_indices_save))
# np.save("save_val_indices.npy",np.array(val_indices_save))

for split, (train_indices, val_indices) in enumerate(zip(train_indices_save,val_indices_save)):

    print('\n\n\n-----------SPLIT {} ------------\n'.format(split))
    if split != 2 and split != 1:
        if warm_up_noisy:
            warm_up_train_indices, train_indices = train_indices[:int((len(train_indices))/5)], indices[int((len(train_indices))/5):]
            warm_up_set = Subset(trainset,warm_up_train_indices)
            warm_up_loader = torch.utils.data.DataLoader(dataset=warm_up_set,
                                                   batch_size=100, 
                                                   shuffle=True)
        cat_dog_testset = Subset(trainset,val_indices)
        cat_dog_trainset = Subset(trainset,train_indices)

        if warm_up_noisy:
            cat_dog_trainset.dataset.add_noise(train_indices,0.2)

        train_loader = torch.utils.data.DataLoader(dataset=cat_dog_trainset,
                                                   batch_size=100, 
                                                   shuffle=True)


        test_loader = torch.utils.data.DataLoader(dataset=cat_dog_testset,
                                                  batch_size=100, 
                                                  shuffle=False)
        holdout_loader = torch.utils.data.DataLoader(dataset=holdoutset,
                                                  batch_size=100, 
                                                  shuffle=False)

        model = ResNet(ResidualBlock, [2, 2, 2],num_classes=2).to(device)
        print(model)

        # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        n_train = 0


        if not load_predictor_model:
            print("Normaaaaal")

            #### log files for multiple runs are NOT overwritten


            log_dir = '2_normal_loss_cv_2_class' + '/'
            if not os.path.exists(log_dir):
                  os.makedirs(log_dir)
            log_dir += str(split) + '/'
            if not os.path.exists(log_dir):
                  os.makedirs(log_dir)
            # self.model = w #TODO

            #### get number of log files in log directory
            run_num = 0
            current_num_files = next(os.walk(log_dir))[2]
            run_num = len(current_num_files)

            #### create new log file for each run 
            log_f_name_train = log_dir + '/Train_normal_' + "_log_" + str(run_num) +"_"+ str(split)+ ".csv"

            log_f_name_val = log_dir + '/Valid_normal_' +  "_log_" + str(run_num) + "_"+ str(split)+ ".csv"



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
                for i, (images, labels, if_noisy) in enumerate(train_loader):
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
                        for images, labels,if_noisy in test_loader:
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
                        torch.save(model.state_dict(), log_dir + 'predictor_resnet18_'+str(split)+'.ckpt')
                    log_f_valid.write('{},{},{}\n'.format(number_of_total_images,epoch, correct / total))
                    log_f_valid.flush()
        else:
            # model.load_state_dict(torch.load('predictor_resnet18.ckpt'))
            pass
        if warm_up_noisy:

            print("warm uuup")
            # Train the model
            total_step = len(train_loader)
            curr_lr = learning_rate
            best_performance = 0

            number_of_total_images = 0
            for epoch in range(num_epochs):
                correct = 0
                total = 0
                for i, (images, labels, if_noisy) in enumerate(train_loader):
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


                # Decay learning rate
                if (epoch+1) % 20 == 0:
                    curr_lr /= 3
                    update_lr(optimizer, curr_lr)

                if (epoch+1) % save_interval ==  0:
                    model.eval()
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for images, labels,if_noisy in test_loader:
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
        else:
            # model.load_state_dict(torch.load('predictor_resnet18.ckpt'))
            pass

        # actor = Actor(2)
        # critic = Critic()
        # # # tensorboard = make_tensorboard()
        # interface=GAInterface(cat_dog_trainset, cat_dog_testset,10,0.2,model,device,"GA_logs_cv/"+str(split))
        # interface=PPOInterface(cat_dog_trainset, cat_dog_testset,actor,critic,model,device,"PPO_logs_noisy_2_class_cv/"+str(split))
        # interface.train(150)

        # # print(p.get_controller_preds_on_holdout())


        # tensorboard.writer.add_scalar('Loss/Training', train_loss.item(), epoch)
        # tensorboard.writer.add_scalar('Accuracy/Training', train_accuracy, epoch)
