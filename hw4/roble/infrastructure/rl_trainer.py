from collections import OrderedDict
import pickle
import os
import sys
import time
import gym
from gym import wrappers
import numpy as np
import torch

from hw2.roble.infrastructure.rl_trainer import RL_Trainer

from hw4.roble.infrastructure import pytorch_util as ptu
from hw4.roble.infrastructure import utils
from hw4.roble.infrastructure.logger import Logger
from hw3.roble.agents.dqn_agent import DQNAgent
from hw3.roble.agents.ddpg_agent import DDPGAgent
from hw3.roble.agents.td3_agent import TD3Agent
from hw3.roble.agents.sac_agent import SACAgent
from hw4.roble.agents.pg_agent import PGAgent

from hw3.roble.infrastructure.dqn_utils import (
        get_wrapper_by_name
)
from hw4.roble.envs.ant.create_maze_env import create_maze_env
from hw4.roble.envs.reacher.reacher_env import create_reacher_env
from hw4.roble.infrastructure.gclr_wrapper import GoalConditionedEnv, GoalConditionedEnvV2
from hw4.roble.infrastructure.hrl_wrapper import HRLWrapper
# how many rollouts to save as videos
MAX_NVIDEO = 1
MAX_VIDEO_LEN = 40 # we overwrite this in the code below

class RL_Trainer(RL_Trainer):

    def __init__(self, params, agent_class = None):

        #############
        ## INIT
        #############
        # Inherit from hw1 RL_Trainer
        super().__init__(params, agent_class)
        

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logging']['logdir'])

        # Set random seeds
        seed = self.params['logging']['random_seed']
        np.random.seed(seed)
        torch.manual_seed(seed)

        if 'env_wrappers' in self.params:
            # These operations are currently only for Atari envs
            self.env = wrappers.Monitor(
                self.env,
                os.path.join(self.params['logging']['logdir'], "gym"),
                force=True,
                video_callable=(None if self.params['logging']['video_log_freq'] > 0 else False),
            )
            self.env = params['env_wrappers'](self.env)
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')
        if 'non_atari_colab_env' in self.params and self.params['logging']['video_log_freq'] > 0:
            self.env = wrappers.Monitor(
                self.env,
                os.path.join(self.params['logging']['logdir'], "gym"),
                force=True,
                video_callable=(None if self.params['logging']['video_log_freq'] > 0 else False),
            )
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')

        self.env.seed(seed)


        #############
        ## AGENT
        #############

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = 1

        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['logging']['video_log_freq'] == 0 and self.params['logging']['video_log_freq'] != -1:
                self.logvideo = True
            else:
                self.logvideo = False

            # decide if metrics should be logged
            if self.params['logging']['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['logging']['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # collect trajectories, to be used for training
            if isinstance(self.agent, DQNAgent) or isinstance(self.agent, DDPGAgent):
                # only perform an env step and add to replay buffer for DQN and DDPG
                self.agent.step_env()
                envsteps_this_batch = 1
                train_video_paths = None
                paths = None
            else:
                use_batchsize = self.params['alg']['batch_size']
                if itr==0:
                    use_batchsize = self.params['alg']['batch_size_initial']
                paths, envsteps_this_batch, train_video_paths = (
                    self.collect_training_trajectories(
                        itr, initial_expertdata, collect_policy, use_batchsize)
                )

            self.total_envsteps += envsteps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr>=start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # add collected data to replay buffer
            
            # print("collected ", len(paths[0]), " trajectories")
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")
            all_logs = self.train_agent()
    

            # log/save
            if self.logvideo or self.logmetrics:
                # perform logging
                print('\nBeginning logging procedure...')

                self.perform_logging(itr, paths, eval_policy, train_video_paths, all_logs)

                if self.params['logging']['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logging']['logdir'], itr))
            if  self.params['alg']['on_policy']:       
                self.agent.clear_mem()


    ####################################
    ####################################
        
    def create_env(self, env_name):
        # Make the gym environment
        if self.params['env']['env_name'] == 'antmaze':
            env = create_maze_env('AntMaze')
        elif self.params['env']['env_name'] == 'reacher':
            env = create_reacher_env()
        else:
            env = gym.make(env_name)
            
                # Call your goal conditioned wrapper here (You can modify arguments depending on your implementation)
        if self.params['env']['task_name'] == 'gcrl':
            env = GoalConditionedEnv(env)     
        elif self.params['env']['task_name'] == 'gcrl_v2':
            env = GoalConditionedEnvV2(env)
        elif self.params['env']['task_name'] == 'hrl':
            env = HRLWrapper(env)
        else:
            pass
        
        return env

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):
        
        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, 
                                eval_policy, self.params['alg']['eval_batch_size'], 
                                self.params['env']['max_episode_length'])

        # save eval rollouts as videos in tensorboard event file
        if self.log_video:
            if train_video_paths is not None:
                #save train/eval videos
                print('\nSaving train rollouts as videos...')
                self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            else:
                print('\nCollecting video rollouts eval')
                eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)
                print('\nSaving eval rollouts as videos...')
                self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        #######################

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]
            eval_avg_rew = [eval_path["reward"].mean() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReward"] = np.mean(eval_avg_rew)
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            
            logs["Critic_Loss"] = np.mean(np.array([all_log["Critic_Loss"].mean() for all_log in all_logs]))
            
            last_log = all_logs[-1]
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return
            
            

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            self.logger.log_file(itr, logs)
            print('Done logging...\n\n')

            self.logger.flush()