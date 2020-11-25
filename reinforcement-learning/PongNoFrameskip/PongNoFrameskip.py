import numpy as np
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym, suite_atari
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step 

# Creation of training environments and evaluation environments
ENV_NAME = 'PongNoFrameskip-v4'
env = suite_atari.load(
    ENV_NAME,
    gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
train_py_env = suite_atari.load(
    ENV_NAME,
    gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
eval_py_env = suite_atari.load(
    ENV_NAME,
    gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

class Norm_pixel(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs/255

# Building Multilayered Neural Networks (Q Networks)
fc_layer_params = (512,)
conv_layer_params = ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))
q_net = q_network.QNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            preprocessing_layers=Norm_pixel(),
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params)
optimizer = tf.compat.v1.train.RMSPropOptimizer(
    learning_rate=2.5e-4,
    decay=0.95,
    momentum=0.0,
    epsilon=1e-2)

time_step_spec = time_step.time_step_spec(train_env.observation_spec())
action_spec = tensor_spec.from_spec(train_env.action_spec())

# Creating a DQN Agent
tf_agent = dqn_agent.DqnAgent(
    time_step_spec,
    action_spec,
    q_network=q_net,
    optimizer=optimizer)
tf_agent.initialize()
# Building a history
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
   data_spec=tf_agent.collect_data_spec,
   batch_size=train_env.batch_size,
   max_length=1_000) # 1_000_000


# Function to calculate the evaluated value
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

# Function to collect and store the experience in the replay buffer
def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)

# First execute a random strategy and then accumulate in the replay buffer
initial_collect_steps = 10 
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
for _ in range(initial_collect_steps):
    collect_step(train_env, random_policy)
dataset = replay_buffer.as_dataset(
   num_parallel_calls=3, sample_batch_size=64, num_steps=2).prefetch(3)
iterator = iter(dataset)

num_eval_episodes = 1 
log_interval = 10 
eval_interval = 50 
num_iterations = 50 
tf_agent.train = common.function(tf_agent.train)
returns = []
for step in range(1, num_iterations+1):
    # Add to the replay buffer the experience gained 
    # by the agent interacting with the environment
    collect_step(train_env, tf_agent.collect_policy)
    # Learning to extract the experience from the replay buffer
    experience, unused_info = next(iterator)
    train_loss = tf_agent.train(experience)
    # Outputs the loss of the model according to the number of steps
    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))
    # Evaluate (test) the model according to the number of steps.
    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

        