import gym
import numpy as np
from gym import spaces
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader,Subset
import os
import pandas as pd 


class TaskAmenability(gym.Env):
    
    def __init__(self, x_train, x_val, task_predictor,device,log_dir,tensorboard=None):
        
        self.x_train = x_train
        self.x_val = x_val
        self.device = device
        self.indexes_of_seen_data = np.zeros(len(self.x_train))
        self.batch_indices = None
        # self.tensorboard = tensorboard
                
        self.img_shape = self.x_train.__getitem__(0)[0].shape
        
        
        self.task_predictor = task_predictor
        
        
        self.controller_batch_size = 1000
        self.task_evaluate_batch_size = 100
        self.task_predictor_batch_size = 100
        self.epochs_per_batch = 50
        self.n_train = 0
        
        self.x_val_loader = torch.utils.data.DataLoader(dataset=self.x_val,
                                                        batch_size=self.task_evaluate_batch_size, 
                                                        shuffle=False)
        
        self.x_train_loader = None
        
        
        
        self.num_val = len(self.x_val)
        
        self.observation_space =  spaces.Box(low=0, high=1, shape=self.img_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        
        self.actions_list = []
        self.val_metric_list = [0.5]*10
        
        self.sample_num_count = 0
        self.total_epoch = 0
        self.total_num_seen_data = 0
        self.total_num_choosen_data = 0
        

        self.criterion = nn.CrossEntropyLoss()
        
        
        #### log files for multiple runs are NOT overwritten
        
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
              os.makedirs(self.log_dir)

        self.log_dir = self.log_dir + '/' + 'rl_loss' + '/'
        if not os.path.exists(self.log_dir):
              os.makedirs(self.log_dir)
        # self.model = w #TODO
        
        #### get number of log files in log directory
        self.run_num = 0
        self.current_num_files = next(os.walk(self.log_dir))[2]
        self.run_num = len(self.current_num_files)
        
        #### create new log file for each run 
        self.log_f_name_train = self.log_dir + '/Train_' + 'rl' + "_log_" + str(self.run_num) + ".csv"
        
        self.log_f_name_val = self.log_dir + '/Valid_' + 'rl' + "_log_" + str(self.run_num) + ".csv"


        
        # logging file
        self.log_f_train = open(self.log_f_name_train,"w+")
        self.log_f_train.write('num_trained_data,epoch,total_num_choosen_data,loss,acc\n')
        
                # logging file
        self.log_f_valid = open(self.log_f_name_val,"w+")
        self.log_f_valid.write('num_trained_data,epoch,total_num_choosen_data,acc\n')
        
        
        self.best_valid_metric = 0
        
        #### get number of log files in save_path directory
        self.saving_path = log_dir+"/checkpoints/"
        if not os.path.exists(self.saving_path):
              os.makedirs(self.saving_path)
        self.run_num = 0
        self.current_num_files = next(os.walk(self.saving_path))[2]
        self.run_num = len(self.current_num_files)
        self.saving_path = self.saving_path + "/predictor_resnet_18_" +str(self.run_num) + ".ckpt"
        print(self.saving_path)
        
        self.seen_indices_path = log_dir+"/seen_indices"
        if not os.path.exists(self.seen_indices_path):
              os.makedirs(self.seen_indices_path)
        self.run_num = 0
        self.current_num_files = next(os.walk(self.seen_indices_path))[2]
        self.run_num = len(self.current_num_files)
        self.seen_indices_path = self.seen_indices_path + "/df" +str(self.run_num) +".csv"
        print(self.seen_indices_path)
        
        
        self.portion_of_none_noisy_selection = 0
        self.portion_of_noisy_reduction = 0
        
    def get_batch(self):
        shuffle_inds = np.random.permutation(len(self.x_train))
        self.x_train = Subset(self.x_train,shuffle_inds)
        self.batch_indices = shuffle_inds[0:self.controller_batch_size]
        return Subset(self.x_train,np.arange(self.controller_batch_size))

    def compute_moving_avg(self):
        self.val_metric_list = self.val_metric_list[-10:]
        moving_avg = np.mean(self.val_metric_list)
        return moving_avg
    
    def select_samples(self, actions_list):
        actions_list = np.clip(actions_list, 0, 1)
        selection_vector = np.random.binomial(1, actions_list)
        logical_inds = []
        selected = 0
        none_noisy_selected = 0
        none_selected = 0
        noisy_none_selected = 0
        for i in range(len(selection_vector)):
            if selection_vector[i] == 1:
                logical_inds.append(i)
                self.indexes_of_seen_data[self.batch_indices[i]] = 1
                selected += 1
                if self.x_train[self.batch_indices[i]][2] == 1:
                    none_noisy_selected += 1
            else:
                none_selected += 1
                if self.x_train[self.batch_indices[i]][2] == 0:
                    noisy_none_selected += 1
                    
                    
        self.portion_of_none_noisy_selection = none_noisy_selected/selected
        self.portion_of_noisy_reduction = noisy_none_selected/none_selected
                
        return Subset(self.x_train_batch,logical_inds)
    
    
    def update_lr(self,optimizer, lr):    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return
    
    def get_val_acc_vec(self):
        self.task_predictor.eval()
        with torch.no_grad():
            correct = []
            # total = 0
            for images, labels,if_noisy in self.x_val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.task_predictor(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).cpu()
        return np.array(correct)
    
    def train_predictor(self,train_data, mode = 'PPO'):
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=self.task_predictor_batch_size, shuffle=True)
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.task_predictor.parameters(), lr=self.learning_rate)
        curr_lr = self.learning_rate
        self.task_predictor.train()
        for epoch in range(self.epochs_per_batch):
            correct = []
            self.total_epoch += 1
            for i, (images, labels,if_noisy) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.total_num_seen_data += len(labels)
                # Forward pass
                outputs = self.task_predictor(images)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).cpu()
                
                # Backward and optimizea
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if (self.n_train) % 500 == 0:
                    pass
                    # self.tensorboard.writer.add_scalar('Train/loss_'+mode, loss.item(), self.n_train)
                    
            train_metric = np.mean(np.multiply(np.array(correct),1))
            if (epoch+1) % 10 == 0:
                self.log_f_train.write('{},{},{},{},{}\n'.format(self.total_num_seen_data,self.total_epoch, sum(self.indexes_of_seen_data),
                                                              loss.item(),np.mean(np.multiply(np.array(correct),1))))
                self.log_f_train.flush()
                    # self.tensorboard.writer.add_scalar('Train/Acc_'+mode, train_metric, self.n_train)
            # Decay learning rate
            if (epoch+1) % 10 == 0:
                curr_lr /= 3
                self.update_lr(self.optimizer, curr_lr)


    def step(self, action):
        self.actions_list.append(action)
        self.sample_num_count += 1
        
        # print(self.sample_num_count)
        
        if self.sample_num_count < self.controller_batch_size:
            reward = 0
            done = False
            return torch.unsqueeze(self.x_train_batch.__getitem__(self.sample_num_count)[0],axis = 0), reward, done, {}
        
        else:
            x_train_selected = self.select_samples(self.actions_list[:self.controller_batch_size])
            if len(x_train_selected) < 1:
                reward = -1
                done = True
            else:
                moving_avg = self.compute_moving_avg()
                
                self.train_predictor(x_train_selected,'PPO')
                
                val_acc_vec = self.get_val_acc_vec()
#                 print(val_acc_vec.shape,len(self.actions_list),self.controller_batch_size)
#                 val_sel_vec = self.actions_list[self.controller_batch_size:]
#                 print(len(self.actions_list))
#                 val_sel_vec_normalised = np.array(val_sel_vec) / np.mean(val_sel_vec)
#                 print(val_sel_vec_normalised.shape)
                
                val_metric = np.mean(np.multiply(np.array(val_acc_vec),1))
                if val_metric > self.best_valid_metric:
                    self.best_valid_metric = val_metric
                    torch.save(self.task_predictor.state_dict(), self.saving_path)
                pd.DataFrame(self.indexes_of_seen_data).to_csv(self.seen_indices_path)
                # self.tensorboard.writer.add_scalar('Val/Acc_PPO', val_metric, self.total_step)
                
                
                self.log_f_valid.write('{},{},{},{}\n'.format(self.total_num_seen_data,self.total_epoch, sum(self.indexes_of_seen_data),val_metric))
                self.log_f_valid.flush()
                self.val_metric_list.append(val_metric)
                reward = val_metric - moving_avg
                # print(reward)
                done = True
            return np.random.rand(self.img_shape[0], self.img_shape[1], self.img_shape[2]), reward, done, {}
        
    def reset(self):
        self.x_train_batch = self.get_batch()
        self.actions_list = []
        self.sample_num_count = 0

        return torch.unsqueeze(self.x_train_batch.__getitem__(self.sample_num_count)[0],axis = 0)
    
    def save_task_predictor(self, task_predictor_save_path):
        self.task_predictor.save(task_predictor_save_path)
        
#     def compute_random(self):
#         actions_list = np.random.randint(2, size=self.controller_batch_size)
#         x_train_selected = self.select_samples(actions_list)
        
#         moving_avg = self.compute_moving_avg()
                
#         self.train_predictor(x_train_selected,'Rand')

#         val_acc_vec = self.get_val_acc_vec()
# #                 print(val_acc_vec.shape,len(self.actions_list),self.controller_batch_size)
# #                 val_sel_vec = self.actions_list[self.controller_batch_size:]
# #                 print(len(self.actions_list))
# #                 val_sel_vec_normalised = np.array(val_sel_vec) / np.mean(val_sel_vec)
# #                 print(val_sel_vec_normalised.shape)

#         val_metric = np.mean(np.multiply(np.array(val_acc_vec),1))
#         self.tensorboard.writer.add_scalar('Val/Acc_Rand', val_metric, self.total_step)
#         self.val_metric_list.append(val_metric)
#         reward = val_metric - moving_avg
#         done = True
#     return

#     def compute_whole(self):
#         moving_avg = self.compute_moving_avg()
                
#         self.train_predictor(x_train_selected,'Rand')

#         val_acc_vec = self.get_val_acc_vec()
# #                 print(val_acc_vec.shape,len(self.actions_list),self.controller_batch_size)
# #                 val_sel_vec = self.actions_list[self.controller_batch_size:]
# #                 print(len(self.actions_list))
# #                 val_sel_vec_normalised = np.array(val_sel_vec) / np.mean(val_sel_vec)
# #                 print(val_sel_vec_normalised.shape)

#         val_metric = np.mean(np.multiply(np.array(val_acc_vec),1))
#         self.tensorboard.writer.add_scalar('Val/Acc_Rand', val_metric, self.total_step)
#         self.val_metric_list.append(val_metric)
#         reward = val_metric - moving_avg
#         done = True
#     return

