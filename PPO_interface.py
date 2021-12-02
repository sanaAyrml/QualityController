from TaskAmenability import TaskAmenability
import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np

import gym
from PPO import PPO
import matplotlib.pyplot as plt 

class PPOInterface():
    def __init__(self, x_train, x_val,actor,critic,  task_predictor,device,log_dir,tensorboard = None, load_models=False, controller_save_path=None, task_predictor_save_path=None):
        self.device = device
        # self.tensorboard = tensorboard
        

        # self.rewards = []
        # self.times = []
        
        if load_models:
            self.task_predictor = task_predictor
            print("1")
            task_predictor.load_state_dict(torch.load(task_predictor_save_path))
            print("2")
        else:
            self.task_predictor = task_predictor
        
        def make_env():
            return TaskAmenability(x_train, x_val, task_predictor, device,log_dir,tensorboard)
        
        self.env = make_env()

        def get_from_env(env, parameter):
            return env.get_attr(parameter)[0]

        # self.n_rollout_steps = self.env.controller_batch_size + len(self.env.x_val) # number of steps per episode (controller_batch_size + val_set_len) multiply by an integer to do multiple episodes before controller update
        
        self.actor = actor
        self.critic = critic
            
        
        
        ######################################################################################################################
        
        self.has_continuous_action_space = False

        self.max_ep_len = self.env.controller_batch_size                    # max timesteps in one episode
        self.max_training_timesteps = int(1e5)   # break training loop if timeteps > max_training_timesteps

        self.print_freq = self.max_ep_len * 2     # print avg reward in the interval (in num timesteps)
        self.log_freq = self.max_ep_len * 2       # log avg reward in the interval (in num timesteps)
        self.save_model_freq = int(3e4)      # save model frequency (in num timesteps)

        self.action_std = 0.6
        self.action_std_decay_freq = self.max_ep_len * 5
        self.action_std_decay_rate = 0.01
        self.min_action_std = 0.1

        self.update_timestep = self.max_ep_len * 2      # update policy every n timesteps
        self.K_epochs = 40               # update policy for K epochs
        self.eps_clip = 0.2              # clip parameter for PPO
        self.gamma = 0.99                # discount factor

        self.lr_actor = 0.0003       # learning rate for actor network
        self.lr_critic = 0.001       # learning rate for critic network

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
        self.log_f_name = self.log_dir + '/PPO_' + 'rl' + "_log_" + str(self.run_num) + ".csv"

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
        
        self.checkpoint_path = self.directory + "PPO_{}_{}_{}.pth".format('rl', self.random_seed, self.run_num)
        print("save checkpoint path : " + self.checkpoint_path)
        
            
        
        ################# training procedure ################

        # initialize a PPO agent
        self.ppo_agent = PPO(self.actor, self.critic, self.lr_actor, 
                        self.lr_critic, self.gamma, self.K_epochs, 
                        self.eps_clip, self.has_continuous_action_space, self.device,self.action_std)
        
        if load_models:
            self.ppo_agent.load(controller_save_path)


    def train(self, num_episodes):
        self.ppo_agent.policy.actor.train()
        self.ppo_agent.policy_old.actor.train()
        time_steps = int(num_episodes*self.max_ep_len)
        
        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print("============================================================================================")


        # logging file
        log_f = open(self.log_f_name,"w+")
        log_f.write('episode,timestep,reward,portion_of_none_noisy_selection,portion_of_noisy_reduction\n')


        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0
        
        max_training_timesteps = int(num_episodes*self.max_ep_len)
        
        print(f'Training started for {num_episodes} episodes:',max_training_timesteps)


        # training loop
        while time_step <= max_training_timesteps:
            
            state = self.env.reset()
            current_ep_reward = 0

            for t in range(1, self.max_ep_len+1):

                # select action with policy
                action = self.ppo_agent.select_action(state)
                state, reward, done, _ = self.env.step(action)
                log_done = 0
                if done:
                    pass
                    # self.tensorboard.writer.add_scalar('Controller/reward', reward , time_step)



                # saving reward and is_terminals
                self.ppo_agent.buffer.rewards.append(reward)
                self.ppo_agent.buffer.is_terminals.append(done)

                time_step +=1
                current_ep_reward += reward
                

                # update PPO agent
                if time_step % self.update_timestep == 0:
                    self.ppo_agent.update()


                # if continuous action space; then decay action std of ouput action distribution
                if self.has_continuous_action_space and time_step % self.action_std_decay_freq == 0:
                    self.ppo_agent.decay_action_std(self.action_std_decay_rate, self.min_action_std)

                # log in logging file
                if time_step % self.log_freq == 0:

                    # log average reward till last episode
                    if log_running_episodes != 0:
                        log_avg_reward = log_running_reward / log_running_episodes
                    else:
                        log_avg_reward = log_running_reward
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{},{},{}\n'.format(i_episode, time_step, log_avg_reward, self.env.portion_of_none_noisy_selection,
                                                         self.env.portion_of_noisy_reduction))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0


                # printing average reward
                if time_step % self.print_freq == 0:

                    # print average reward till last episode
                    if print_running_episodes != 0:
                        print_avg_reward = print_running_reward / print_running_episodes
                    else:
                        print_avg_reward = print_running_reward
                    print_avg_reward = round(print_avg_reward, 2)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % self.save_model_freq == 1:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + self.checkpoint_path)
                    self.ppo_agent.save(self.checkpoint_path)
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")

                # break; if the episode is over
                if done:
                    break


            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1


        log_f.close()
        self.env.close()


        
    def get_controller_preds_on_holdout(self, x_holdout_loader):
        
        actions = []
        action_logprobs = []
        self.ppo_agent.policy.actor.eval()
        with torch.no_grad():
            for states, labels,if_noisy in x_holdout_loader:
                action , action_logprob = self.ppo_agent.policy.act(states.to(self.device))
                actions += list(action.detach().cpu().numpy())
                action_logprobs += list(action_logprob.detach().cpu().numpy())
        return np.array(actions),np.array(action_logprobs)
            


        
#     def save(self, controller_save_path, task_predictor_save_path):
#         self.model.save(controller_save_path)
#         task_predictor_copy = self.env.get_attr('task_predictor')[0]
#         task_predictor_copy.save(task_predictor_save_path)
        
#     def load(self, save_path):
#         self.model = PPO2.load(save_path)
#         self.model.set_env(self.env)
        
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

