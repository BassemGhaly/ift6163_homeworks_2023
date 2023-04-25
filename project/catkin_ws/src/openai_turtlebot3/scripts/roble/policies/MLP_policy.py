import abc
import itertools
import numpy as np
import torch
import roble.util.class_util as classu

from roble.infrastructure import pytorch_util as ptu
from hw1.roble.policies.base_policy import BasePolicy
from roble.infrastructure.utils import normalize
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions

class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    @classu.hidden_member_initialize
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 deterministic=False,
                 learn_policy_std=True,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars

        if self._discrete:
            self._logits_na = ptu.build_mlp(input_size=self._ob_dim,
                                           output_size=self._ac_dim,
                                           n_layers=self._n_layers,
                                           size=self._size)
            self._logits_na.to(ptu.device)
            self._mean_net = None
            self._logstd = None
            self._optimizer = optim.Adam(self._logits_na.parameters(),
                                        self._learning_rate)
        else:
            self._logits_na = None
            self._mean_net = ptu.build_mlp(input_size=self._ob_dim, 
                                      output_size=self._ac_dim,
                                      n_layers=self._n_layers, size=self._size)
            self._mean_net.to(ptu.device)
            if self._deterministic:
                self._optimizer = optim.Adam(
                    itertools.chain(self._mean_net.parameters()),
                    self._learning_rate
                )
            else:
                self._std = nn.Parameter(
                    torch.ones(self._ac_dim, dtype=torch.float32, device=ptu.device) * 0.15
                )
                self._std.to(ptu.device)
                if self._learn_policy_std:
                    self._optimizer = optim.Adam(
                        itertools.chain([self._std], self._mean_net.parameters()),
                        self._learning_rate
                    )
                else:
                    self._optimizer = optim.Adam(
                        itertools.chain(self._mean_net.parameters()),
                        self._learning_rate
                    )
                

        if nn_baseline:
            self._baseline = ptu.build_mlp(
                input_size=self._ob_dim,
                output_size=1,
                n_layers=self._n_layers,
                size=self._size,
            )
            self._baseline.to(ptu.device)
            self._baseline_optimizer = optim.Adam(
                self._baseline.parameters(),
                self._learning_rate,
            )
        else:
            self._baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # DONE return the action that the policy prescribes
        obs = ptu.from_numpy(observation.astype(np.float32))
        actions = None

        if self._deterministic:
            return ptu.to_numpy(self(obs))

        if self._discrete:
            distrib = self(obs)
            return ptu.to_numpy(distrib.sample())

        return ptu.to_numpy(self(obs).rsample())

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self._discrete:
            logits = self._logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            if self._deterministic:
                action_distribution = self._mean_net(observation)
            else:
                batch_mean = self._mean_net(observation)
                scale_tril = torch.diag(self._std)
                # print ("scale_tril:", scale_tril)
                batch_dim = batch_mean.shape[0]
                batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
                action_distribution = distributions.MultivariateNormal(
                    batch_mean,
                    scale_tril=batch_scale_tril,
                )
        return action_distribution

class ConcatMLP(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self._dim)
        return super().forward(flat_inputs, **kwargs)

class MLPPolicyDeterministic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, **kwargs):
        kwargs['deterministic'] = True
        super().__init__(*args, **kwargs)
        
    def update(self, observations, q_fun):
        self._optimizer.zero_grad()
        obs = ptu.from_numpy(observations)
        ac_na = self.forward(obs)
        qa_values = q_fun.q_net(obs, ac_na)
        loss = - qa_values.mean()
        loss.backward()
        self._optimizer.step()
        return loss.item()
    
class MLPPolicyStochastic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, entropy_coeff, *args, **kwargs):
        kwargs['deterministic'] = False
        super().__init__(*args, **kwargs)
        self.entropy_coeff = entropy_coeff

    def get_action(self, obs: np.ndarray) -> np.ndarray:

        action = self.forward(ptu.from_numpy(obs)).rsample()
        
        return ptu.to_numpy(action)
        
    def update(self, observations, q_fun):

        self._optimizer.zero_grad()
        obs = ptu.from_numpy(observations)
        normal = self.forward(obs)
        actions = normal.rsample()
        log_prob = normal.log_prob(actions)
        qa_values = q_fun.q_net(obs, actions).detach()
        qa_values_logprob = qa_values - self.entropy_coeff * log_prob
        loss = - qa_values_logprob.mean()
        
        loss.backward()
        self._optimizer.step()
        return loss.item()
    
class MLPPolicyPG(MLPPolicy):
    
    @classu.hidden_member_initialize
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        action_distribution = self(observations)
        loss = - action_distribution.log_prob(actions) * advantages
        loss = loss.mean()
    
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        train_log = {
            'Training_Loss': ptu.to_numpy(loss),
        }
        if self._nn_baseline:
            targets_n = normalize(q_values, np.mean(q_values), np.std(q_values))
            targets_n = ptu.from_numpy(targets_n)
            baseline_predictions = self._baseline(observations).squeeze()
            assert baseline_predictions.dim() == baseline_predictions.dim()
    
            baseline_loss = F.mse_loss(baseline_predictions, targets_n)
            self._baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self._baseline_optimizer.step()
            train_log["Critic_Loss"] = ptu.to_numpy(baseline_loss)
        else:
            train_log["Critic_Loss"] = ptu.to_numpy(0)

        return train_log

    def run_baseline_prediction(self, observations):
        """
            Helper function that converts `observations` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `observations`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        observations = ptu.from_numpy(observations)
        pred = self._baseline(observations)
        return ptu.to_numpy(pred.squeeze())