"""
Q-Learning Agent for OpenAI Gymnasium Taxi Environment

This module implements a Q-learning reinforcement learning agent to solve the
classic Taxi-v3 environment from OpenAI Gymnasium. The taxi agent learns to
navigate a 5x5 grid world to pick up and drop off passengers at designated
locations while maximizing cumulative rewards.

The Q-learning algorithm uses temporal difference learning to update a Q-table
that maps state-action pairs to expected future rewards. The agent follows an
epsilon-greedy policy, balancing exploration of new actions with exploitation
of learned optimal actions.

Environment Details:
    - State space: 500 discrete states (taxi position, passenger location, destination)
    - Action space: 6 discrete actions (north, south, east, west, pickup, dropoff)
    - Rewards: +20 for successful dropoff, -10 for illegal pickup/dropoff, -1 per step

Classes:
    QlearningConfig: Configuration parameters for the Q-learning algorithm

Functions:
    run: Main training loop that runs Q-learning for specified episodes
    choose_action: Epsilon-greedy action selection based on current Q-values

Global Variables:
    _env: Gymnasium Taxi environment instance
    _q_table: Q-value table for state-action pairs
    _q_learning_config: Configuration instance with hyperparameters

Example:
    Run the Q-learning training:

    $ python train.py

    Or import and use programmatically:

    >>> from train import run, QlearningConfig
    >>> config = QlearningConfig()
    >>> config.epsilon = 0.1  # Reduce exploration
    >>> run(n_episodes=50000)

Author: vnniciusg
Date: July 19, 2025
"""

from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from loguru import logger


def setup_env(
    env_name: str = "Taxi-v3",
    num_eval_episodes: int = 10_000,
    render_mode: Literal["rgb_array", "human", "ansi"] = "rgb_array",
    training_period: int = 250,
):
    """
    Set up the Gymnasium environment with video recording and statistics tracking.

    This function creates a Taxi environment wrapped with monitoring capabilities
    to record videos of agent performance and track episode statistics during
    training and evaluation.

    Args:
        env_name (str): Name of the Gymnasium environment to create.
            Default is "Taxi-v3".
        num_eval_episodes (int): Buffer size for episode statistics tracking.
            Determines how many episodes of statistics to keep in memory.
            Default is 100,000.
        render_mode (str): Rendering mode for the environment. Options include:
            - "rgb_array": Returns RGB array for video recording
            - "human": Displays environment in a window
            - "ansi": Text-based rendering
            Default is "rgb_array".
        training_period (int): Record video every 250 episodes.

    Returns:
        gym.Env: Wrapped Gymnasium environment with the following capabilities:
            - Video recording of all episodes
            - Episode statistics tracking (rewards, lengths, etc.)
            - Original Taxi environment functionality

    Note:
        Videos are saved to the "taxi-agent/" directory with the prefix "eval".
        The RecordVideo wrapper will record every episode due to the lambda
        function returning True for all episodes.

    Example:
        >>> env = setup_env(env_name="Taxi", render_mode="human")
        >>> observation, info = env.reset()
        >>> # Environment is now ready with video recording enabled
    """

    env = gym.make(env_name, render_mode=render_mode)
    env = RecordVideo(
        env,
        video_folder="taxi-agent",
        name_prefix="eval",
        episode_trigger=lambda x: x % training_period == 0,
    )
    env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)
    return env


_env = setup_env()
_q_table = np.zeros([_env.observation_space.n, _env.action_space.n])


class QlearningConfig:
    """
    Configuration parameters for Q-learning algorithm.

    Attributes:
        alpha (float): Learning rate that determines how much new information
            overrides old information. Range: [0, 1]. Higher values make the
            agent learn faster but may cause instability.
        gamma (float): Discount factor that determines the importance of future
            rewards. Range: [0, 1]. Values closer to 1 make the agent consider
            long-term rewards more heavily.
        epsilon (float): Exploration rate for epsilon-greedy policy. Range: [0, 1].
            Probability of taking a random action instead of the greedy action.
            Higher values encourage more exploration.

    Examples:
        >>> config = QlearningConfig()
        >>> config.alpha = 0.2  # Increase learning rate
        >>> config.epsilon = 0.1  # Reduce exploration
    """

    alpha: float = 0.1
    gamma: float = 0.7
    epsilon: float = 0.2


_q_learning_config = QlearningConfig()


def run(*, n_episodes: int = 10_000):
    """
    Run Q-learning training for the specified number of episodes.

    This function trains the Q-learning agent by running multiple episodes
    of the Taxi environment. In each episode, the agent learns by updating
    Q-values based on the rewards received for state-action pairs.

    Args:
        n_episodes (int): Number of training episodes to run. Default is 100,000.
    """

    logger.info(f"Starting evaluation for {n_episodes} episodes...")
    logger.info("Videos will be saved to: taxi-agent/")

    for idx in range(n_episodes):
        if idx % 1000 == 0 and idx > 0:
            logger.info(f"Running EPISODE {idx}")

        observation, _ = _env.reset()

        episode_over = False
        total_reward = 0

        while not episode_over:
            action = choose_action(observation=observation)

            next_state, reward, terminated, truncated, _ = _env.step(action)

            total_reward += reward
            episode_over = terminated or truncated

            curr_value = _q_table[observation, action]
            next_max = np.max(_q_table[next_state])

            _q_table[observation, action] = (
                1 - _q_learning_config.alpha
            ) * curr_value + _q_learning_config.alpha * (
                reward + _q_learning_config.gamma * next_max
            )

            observation = next_state

        if idx % 1000 == 0 and idx > 0:
            logger.success(f"Episode {idx} finished! Total Reward: {total_reward}")

    _env.close()
    logger.success("Training completed! Environment closed.")


def choose_action(observation) -> int:
    if np.random.random() < _q_learning_config.epsilon:
        return _env.action_space.sample()

    return np.argmax(_q_table[observation])


if __name__ == "__main__":
    run()
