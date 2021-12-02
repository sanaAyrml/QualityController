from TaskAmenability_GA import TaskAmenability_GA
import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np
from torch.distributions import Bernoulli
from torch.distributions import Normal
import gym
from PPO import PPO
import matplotlib.pyplot as plt 
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from predictor_model import ResNet,ResidualBlock

from GA_Controler import GA_Controler

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.probs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.probs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class GAInterface():
    def __init__(self, x_train, x_val, number_of_initial, survive_percent,  task_predictor,device,log_dir,tensorboard = None, load_models=False, controller_save_path=None, task_predictor_save_path=None):
        self.device = device
        # self.tensorboard = tensorboard
        

        self.number_of_initial = number_of_initial
        self.survive_percent =  survive_percent
        self.population =  []
        
        self.buffer = RolloutBuffer() 
        
        if load_models:
            self.task_predictor = task_predictor
            task_predictor.load_state_dict(torch.load(task_predictor_save_path))
        else:
            self.task_predictor = task_predictor
        

        self.task_predictors = []
        def make_env():
            return TaskAmenability_GA(x_train, x_val, self.task_predictors, device,log_dir,tensorboard)
        
        for i in range(self.number_of_initial):
            self.population.append(GA_Controler(2).to(self.device))
            model = ResNet(ResidualBlock, [2, 2, 2],num_classes=2).to(device)
            model.load_state_dict(task_predictor.state_dict())
            self.task_predictors.append(model)
        self.env = make_env()

        def get_from_env(env, parameter):
            return env.get_attr(parameter)[0]
        
        # print(self.task_predictors)

        # self.n_rollout_steps = self.env.controller_batch_size + len(self.env.x_val) # number of steps per episode (controller_batch_size + val_set_len) multiply by an integer to do multiple episodes before controller update
        
        
        
        ######################################################################################################################
        
        self.batch_size = 5
        self.earning_rate = 0.01
        self.gamma = 0.99
        self.epsilon = 0.0001
        self.mean_every_epoch = 10
        
        self.random_seed = 0         # set random seed if required (0 = no random seed)

        #### log files for multiple runs are NOT overwritten
        
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
              os.makedirs(self.log_dir)

        self.log_dir = self.log_dir + '/' + 'rl' + '/'
        if not os.path.exists(self.log_dir):
              os.makedirs(self.log_dir)
        # self.model = w #TODO
        
        #### get number of log files in log directory
        self.run_num = 0
        self.current_num_files = next(os.walk(self.log_dir))[2]
        self.run_num = len(self.current_num_files)
        
        #### create new log file for each run 
        self.log_f_name = self.log_dir + '/GA_' + 'rl' + "_log_" + str(self.run_num) + ".csv"

        print("current logging run number for " + 'rl' + " : ", self.run_num)
        print("logging at : " + self.log_f_name)
        
        ################### checkpointing ###################

        self.run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

        self.directory = log_dir
        if not os.path.exists(self.directory):
              os.makedirs(self.directory)

        self.directory = self.directory + '/' + 'models' + '/'
        if not os.path.exists(self.directory):
              os.makedirs(self.directory)

        #### get number of log files in log directory
        self.run_num = 0
        self.current_num_files = next(os.walk(self.directory))[2]
        self.run_num = len(self.current_num_files)

        self.checkpoint_path = self.directory + "GA_{}_{}_{}.pth".format('rl', self.random_seed, self.run_num)
        print("save checkpoint path : " + self.checkpoint_path)
        
        self.log_freq  = 1
        self.save_model_freq = 1
        self.print_freq = 1
        
        if load_models:
            premodel =  GA_Controler(2).to(self.device)
            premodel.load_state_dict(torch.load(controller_save_path))
            self.population =  [premodel]
            for itr in range(number_of_initial-1):
                new_model = GA_Controler(2).to(self.device)
                new_model.load_state_dict(premodel.state_dict())
                param_vector = parameters_to_vector(new_model.parameters())
                n_params = len(param_vector)
                noise = self.epsilon*Normal(0, 1).sample_n(n_params)
                param_vector.add_(noise.to(self.device))
                self.population.append(new_model)

        


    def train(self, num_episodes):
        # logging file
        log_f = open(self.log_f_name,"w+")
        log_f.write('episode,max fitness score,portion_of_none_noisy_selection,portion_of_noisy_reduction\n')
        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print("============================================================================================")
        
        
        for e in range(num_episodes):
            fitness_scores = []
            durations_per_iter = []
            for p_n,policy_model in enumerate(self.population):
                state = self.env.reset()
                done = False
                self.buffer.clear()
                episod_reward = []
                while not done:
                    self.buffer.states.append(state)
                    self.buffer.is_terminals.append(done)


                    probs = policy_model(state.to(self.device))
                    m = Bernoulli(probs.squeeze(0)[1])
                    action = m.sample()
                    
                    self.buffer.actions.append(action)
                    self.buffer.probs.append(probs)

                    next_state, reward, done, _ = self.env.step(action.item(),p_n)

                    self.buffer.rewards.append(reward)

                    state = next_state

                # discounted_reward = 0
                # for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):   
                #     print()
                #     discounted_reward = reward + (self.gamma * discounted_reward)
                
                fitness_scores.append(self.buffer.rewards[len(self.buffer.rewards)-1])
                # print("hereee",fitness_scores)

            index_sorted_fitness_scores = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[-1*int(len(fitness_scores)*self.survive_percent):]
            print(index_sorted_fitness_scores)
            new_population = []
            # log in logging file
            if e % self.log_freq == 0:
                # print(index_sorted_fitness_scores)

                log_f.write('{},{},{},{}\n'.format(e, fitness_scores[index_sorted_fitness_scores[0]],
                                                   self.env.portion_of_none_noisy_selection,
                                                   self.env.portion_of_noisy_reduction))
                log_f.flush()



            # printing average reward
            if e % self.print_freq == 0:


                print("Episode : {} \t\t Max fitness score : {}".format(e, fitness_scores[index_sorted_fitness_scores[0]]))


            # save model weights
            if e % self.save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + self.checkpoint_path)
                torch.save(self.population[index_sorted_fitness_scores[0]].state_dict(), self.checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")


            new_population = []
            new_predictors = []
            for f in index_sorted_fitness_scores:
                base_model = self.population[f]
                base_predictor = self.task_predictors[f]
                new_population.append(base_model)
                new_predictors.append(base_predictor)
                for itr in range(int(1/self.survive_percent)-1):
                    new_model = GA_Controler(2).to(self.device)
                    new_predictor = ResNet(ResidualBlock, [2, 2, 2],num_classes=2).to(self.device)
                    new_model.load_state_dict(base_model.state_dict())
                    new_predictor.load_state_dict(base_predictor.state_dict())
                    param_vector = parameters_to_vector(new_model.parameters())
                    n_params = len(param_vector)
                    noise = self.epsilon*Normal(0, 1).sample_n(n_params)
                    param_vector.add_(noise.to(self.device))
                    new_population.append(new_model)
                    new_predictors.append(new_predictor)
            self.population = new_population
            self.task_predictors = new_predictors


        
    def get_controller_preds_on_holdout(self,x_holdout_loader):
        model = self.population[0]
        actions = []
        model.eval()
        with torch.no_grad():
            for states, labels,if_noisy in x_holdout_loader:
                probs = model(states.to(self.device))
                # print(probs)
                
                # m = Bernoulli(probs.squeeze(0)[1])
                # action = m.sample()
                m = Bernoulli(torch.transpose(probs, 0, 1)[1])
                action = m.sample()
                actions += list(action.squeeze().detach().cpu().numpy())
        return np.array(actions)
            


        
    def save(self, controller_save_path, task_predictor_save_path):
        self.model.save(controller_save_path)
        task_predictor_copy = self.env.get_attr('task_predictor')[0]
        task_predictor_copy.save(task_predictor_save_path)
        
    def load(self, save_path):
        self.model = PPO2.load(save_path)
        self.model.set_env(self.env)
        
    def verbose(self):
        ############# print all hyperparameters #############

        print("--------------------------------------------------------------------------------------------")

        print("max training timesteps : ", self.max_training_timesteps)
        print("max timesteps per episode : ", self.max_ep_len)

        print("model saving frequency : " + str(self.save_model_freq) + " timesteps")
        print("log frequency : " + str(self.log_freq) + " timesteps")
        print("printing average reward over episodes in last : " + str(self.print_freq) + " timesteps")

        print("--------------------------------------------------------------------------------------------")

        # print("state space dimension : ", state_dim)
        # print("action space dimension : ", action_dim)

        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            print("Initializing a continuous action space policy")
            print("--------------------------------------------------------------------------------------------")
            print("starting std of action distribution : ", self.action_std)
            print("decay rate of std of action distribution : ", self.action_std_decay_rate)
            print("minimum std of action distribution : ", self.min_action_std)
            print("decay frequency of std of action distribution : " + str(self.action_std_decay_freq) + " timesteps")

        else:
            print("Initializing a discrete action space policy")

        print("--------------------------------------------------------------------------------------------")

        print("PPO update frequency : " + str(self.update_timestep) + " timesteps") 
        print("PPO K epochs : ", self.K_epochs)
        print("PPO epsilon clip : ", self.eps_clip)
        print("discount factor (gamma) : ", self.gamma)

        print("--------------------------------------------------------------------------------------------")

        print("optimizer learning rate actor : ", self.lr_actor)
        print("optimizer learning rate critic : ", self.lr_critic)


        
    # implement saving and loading task predictor


