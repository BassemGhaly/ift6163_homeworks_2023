import numpy as np

from .base_agent import BaseAgent
from hw4.roble.policies.MLP_policy import MLPPolicyPG
from hw4.roble.infrastructure.replay_buffer import ReplayBuffer
from hw4.roble.infrastructure.utils import normalize

class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        print ("self.agent_params: ", self.agent_params)
        self.gamma = self.agent_params['discount']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        if self.gae_lambda == 'None':
            self.gae_lambda = None

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline'],
            learn_policy_std=self.agent_params['learn_policy_std'],
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        q_values = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, q_values, terminals)
        advantages = advantages / np.std(advantages) ## Try to keep the statistics of the advantages standardized
        train_log = self.actor.update(observations, actions, advantages=advantages, q_values=q_values)

        return train_log

    def calculate_q_vals(self, rews_list):
        
        """
            Monte Carlo estimation of the Q function.
        """
        
        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        if not self.reward_to_go:
        
            # For each point (s_t, a_t), associate its value as being the discounted sum of rewards over the full trajectory
            # In other words: value of (s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            q_values = np.concatenate([self._discounted_return(r) for r in rews_list])
        
        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
        
            # For each point (s_t, a_t), associate its value as being the discounted sum of rewards over the full trajectory
            # In other words: value of (s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            q_values = np.concatenate([self._discounted_cumsum(r) for r in rews_list])
        
        return q_values

    def estimate_advantage(self, obs, rews_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert values_unnormalized.ndim == q_values.ndim
            ## Values were trained with standardized q_values, so ensure
                ## that the predictions have the same mean and standard deviation as
                ## the current batch of q_values

            values_normalized = (values_unnormalized - values_unnormalized.mean()) / (values_unnormalized.std() + 1e-8)
            values = values_normalized * np.std(q_values) + np.mean(q_values)

            if self.gae_lambda is not None:
                ## append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                
                ## combine rews_list into a single array
                rews = np.concatenate(rews_list)

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    ended = 1 - terminals[i]
                    delta_t = rews[i] + (self.gamma * values[i+1] * ended) - values[i]
                    advantages[i] = delta_t + (self.gae_lambda * self.gamma * advantages[i+1] * ended) 

                # remove dummy advantage
                advantages = advantages[:-1]
            else:
                advantages = q_values - values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)
        
    def clear_mem(self):
        self.replay_buffer.reset()

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    def save(self, path):
        return self.actor.save(path)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function
    
            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T
    
            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
                note that all entries of this output are equivalent
                because each sum is from 0 to T (and doesnt involve t)
        """
    
        # create a list of indices (t'): from 0 to T
        indices = np.arange(len(rewards))
    
        # create a list where the entry at each index (t') is gamma^(t')
        discounts = self.gamma**indices
    
        # create a list where the entry at each index (t') is gamma^(t') * r_{t'}
        discounted_rewards = discounts * rewards
    
        # scalar: sum_{t'=0}^T gamma^(t') * r_{t'}
        sum_of_discounted_rewards = sum(discounted_rewards)
    
        # list where each entry t contains the same thing
            # it contains sum_{t'=0}^T gamma^t' r_{t'}
        list_of_discounted_returns = np.ones_like(rewards) * sum_of_discounted_rewards
    
        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
    
        all_discounted_cumsums = []
    
        # for loop over steps (t) of the given rollout
        for start_time_index in range(len(rewards)):
    
            # create a list of indices (t'): goes from t to T
            indices = np.arange(start_time_index, len(rewards))
    
            # create a list of indices (t'-t)
            indices_adjusted = indices - start_time_index
    
            # create a list where the entry at each index (t') is gamma^(t'-t)
            discounts = self.gamma**(indices_adjusted) # each entry is gamma^(t'-t)
    
            # create a list where the entry at each index (t') is gamma^(t'-t) * r_{t'}
            discounted_rtg = discounts * rewards[start_time_index:]
    
            # scalar: sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            sum_discounted_rtg = sum(discounted_rtg)
            all_discounted_cumsums.append(sum_discounted_rtg)
    
        list_of_discounted_cumsums = np.array(all_discounted_cumsums)
        return list_of_discounted_cumsums
