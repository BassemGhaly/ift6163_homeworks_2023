#!/usr/bin/env python
import os
import time
import sys
import yaml
print(sys.path)
#import hydra, json
import rospy
import json, hydra
from omegaconf import DictConfig, OmegaConf
from roble.agents.dqn_agent import DQNAgent
from hw3.roble.agents.ddpg_agent import DDPGAgent
from hw3.roble.agents.td3_agent import TD3Agent
from hw3.roble.agents.sac_agent import SACAgent
from roble.agents.pg_agent import PGAgent
from roble.infrastructure.rl_trainer import RL_Trainer
#from omegaconf import DictConfig, OmegaConf
from hw3.roble.infrastructure.dqn_utils import get_env_kwargs

class offpolicy_Trainer(object):

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################
        rospy.init_node('turtlebot2_maze_qlearn', anonymous=True, log_level=rospy.WARN)
        self.params = params
        self.params['alg']['batch_size_initial'] = self.params['alg']['batch_size']

        if self.params['alg']['rl_alg'] == 'dqn':
            agent = DQNAgent
        elif self.params['alg']['rl_alg'] == 'ddpg':
            agent = DDPGAgent
        elif self.params['alg']['rl_alg'] == 'td3':
            agent = TD3Agent    
        elif self.params['alg']['rl_alg'] == 'sac':
            agent = SACAgent
        elif self.params['alg']['rl_alg'] == 'pg':
            agent = PGAgent
        else:
            print("Pick a rl_alg first")
            sys.exit()
        print(self.params)
        print(self.params['alg']['train_batch_size'])

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params , agent_class =  agent)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            self.params['alg']['n_iter'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )

'''
@hydra.main(config_path="conf", config_name="config_hrl")
def my_main(cfg: DictConfig):
    my_app(cfg)
'''
import subprocess
import yaml

def get_ros_params():
    param_list = rospy.get_param_names()
    filtered_params = [param for param in param_list if '/hrlparam/' in param]
    output = {}
    for param in filtered_params:
        param_split = param.split('/')
        namespace = param_split[1]
        key = param_split[-1]
        key1 = param_split[2]
        value = rospy.get_param(param)
        if namespace not in output:
            output[namespace]= {}
        if key1 not in output[namespace]:
            output[namespace][key1] = {}
        output[namespace][key1][key] = value
    return output

def my_app(): 
    
    params = get_ros_params()
    params = params['hrlparam']
    rospy.logwarn("params: " + str(params))

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################


    logdir_prefix = 'project_'  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    '''
    exp_name = logdir_prefix + cfg.env.exp_name + '_' + cfg.env.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, exp_name)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.logging.logdir = logdir
        cfg.logging.exp_name = exp_name

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")
    '''
    ###################
    ### RUN TRAINING
    ###################
    # cfg = OmegaConf.merge(cfg, params)
    trainer = offpolicy_Trainer(params)
    trainer.run_training_loop()

if __name__ == '__main__':
    import os
    rospy.loginfo("Command Dir:", os.getcwd())
    my_app()
