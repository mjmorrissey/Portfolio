from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from copy import deepcopy
import gym
from gym.envs.registration import register
from numpy import average
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import DQNPolicy

from evaluation import Evaluation
from gym_slap.envs.interface_input import Input

import torch as th
from torch import nn
import torch.nn.functional as f

from stable_baselines3 import PPO, DQN, HerReplayBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback


# <editor-fold desc="Custom Agent Policy">
from storage_policies import FurthestOpenLocation, ClosestOpenLocation, LIFO, \
    SinkRelativeClosestAvailableSKU, ClassBasedPopularity, RandomOpenLocation
from output_converters import OutputConverter
from training_logging import TrainingLogger
from use_case import UseCase


class QNet(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the
        features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of
        the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of
        the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super(QNet, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        # Value network layers
        self.fc_v1 = nn.Linear(feature_dim, 128)
        self.fc_v2 = nn.Linear(128, last_layer_dim_vf)

    def forward(self, features: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the
            specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        # value head
        v = f.relu(self.fc_v1(features))
        v = f.relu(self.fc_v2(v))
        return v


class MlpExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 256):
        super(MlpExtractor, self).__init__(observation_space, features_dim)
        input_size = observation_space.shape[0]
        self.fc1 = nn.Linear(input_size, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.fc1(observations)
        return self.relu(x)


class FCDQNPolicy(DQNPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[
                List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):
        super(FCDQNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.q_net = QNet(self.features_dim)


if __name__ == "__main__":
    # <editor-fold desc="DEFINE ENVIRONMENT ARGS AND BUILD ENV">
    # use_case = UseCase(100)
    env_params = {
        'n_rows': 20,
        'n_columns': 20,
        'n_levels': 3,
        'n_agvs': 9,  # 5 # 2,
        'n_skus': 2,
        'generate_orders': True,
        'n_orders': 100,
        'desired_fill_level': 0.1,
        'verbose': False,
        'state_stack_size': 1,
        'resetting': False,
        'order_list': None,
        'initial_pallets_sku_counts': None,
        'initial_pallets_storage_strategy': ClassBasedPopularity(used_for_initialization=True),
        'n_sources': 3,
        'n_sinks': 3,
        'use_case': 10000 # trained on 100
    }
    seeds = [56513, 30200, 28174, 9792, 63446, 81531, 31016, 5161, 8664,
                  12399, 6513, 200, 2174, 792, 3446, 1531, 1016, 161, 664,
                  1239]
    logfile_path = ''
    output_converter = OutputConverter('average_travel_length', 'lane_free_space')
    storage_strategies = [ClassBasedPopularity(retrieval_orders_only=True),
                        ClassBasedPopularity(),
                        RandomOpenLocation(), FurthestOpenLocation(),
                        ClosestOpenLocation()]
    # storage_strategies = []
    retrieval_strategies = [SinkRelativeClosestAvailableSKU()]

    gym.register( id='slap-interface-v0',
                    entry_point='gym_slap.envs:StrategyInterface',
                  kwargs={
                      'environment_parameters': env_params,
                      'seeds': seeds,
                      'logfile_path': logfile_path,
                      'output_converter': output_converter,
                      'selectable_strategies': storage_strategies
                                               + retrieval_strategies})
    env = gym.make('slap-interface-v0')
    # 1. define agent with custom network
    model = DQN(FCDQNPolicy, env, verbose=2,
                learning_starts=2500,
                target_update_interval=500,
                train_freq=100,
                policy_kwargs=dict(
                    features_extractor_class=MlpExtractor,
                    features_extractor_kwargs=dict(features_dim=150)),
                exploration_initial_eps=0.1,
                learning_rate=0.000025,
                exploration_final_eps=0.1,
                batch_size=64,
                buffer_size=1000,
                tensorboard_log="./tensorboard/dqn",)


    # 2. define evaluation environment and evaluation callback
    eval_env = deepcopy(env)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best',
                                n_eval_episodes=5,
                                 #eval_freq=500, n_eval_episodes=5,
                                 deterministic=True, render=False)


    # log model, environment, eval callback
    logger = TrainingLogger(model, env, eval_callback, seeds)
    logger.create_json()

    # 3. fit the model
    model.learn(total_timesteps=int(500000), callback=[eval_callback])


    # 1. load trained model
    model.load('./models/best/best_model.zip', env=None, custom_objects=None)

    # 2. evaluate by comparing with the underlying heuristics
    evaluation = Evaluation(model, env_params, seeds, logfile_path,
                            output_converter, storage_strategies, retrieval_strategies)
    evaluation_results = evaluation.evaluate_model(n_episodes=1)
    logger.add_evaluation(evaluation_results)

