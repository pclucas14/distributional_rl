import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import argparse
import pdb
import gym
import argparse
import os
import tensorboardX

from common.replay_buffer import ReplayBuffer
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from common.logger import Logger
from models import * 
from losses import * 
from utils  import * 

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='cartpole')
parser.add_argument('--num_atoms', type=int, default=51)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--loss', type=str, default='cramer')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--eval_runs', type=int, default=100)
parser.add_argument('--base_dir', type=str, default='experiments/test')
parser.add_argument('--exp_common_log', type=str, default='experiments/common.txt')
parser.add_argument('--no_projection', action='store_true')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

# reproducibility
if args.seed:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

if args.base_dir == 'experiments/test':
    args.base_dir = 'experiments/{}_{}_{}'.format(args.env, args.loss, args.num_atoms)

args_dict = vars(args)

# GPU settings 
USE_CUDA = torch.cuda.is_available()
args_dict['cuda'] = USE_CUDA

# add environment settings
if 'cartpole' in args.env.lower():
    env = gym.make('CartPole-v0')
    args_dict['Vmin'] = 0
    args_dict['Vmax'] = 200
    num_frames = 20000
elif 'acrobot' in args.env.lower():
    env = gym.make('Acrobot-v1')
    args_dict['Vmin'] = -500
    args_dict['Vmax'] = 0
    num_frames = 40000
elif 'pong' in args.env.lower():
    env = make_atari('PongNoFrameskip-v4')
    args_dict['Vmin'] = -20
    args_dict['Vmax'] = 20
    num_frames = 4000000
elif 'breakout' in args.env.lower():
    env = gym.make('Breakout-v4')
    args_dict['Vmin'] = 0
    args_dict['Vmax'] = 350
    num_frames = 4000000
else: 
    raise ValueError('invalid environment')

# more general settings
if args.env.lower() in ['cartpole', 'acrobot']: 
    model = CategoricalDQN
    lr = 4e-4
    replay_buffer_size = 10000
    update_target_every = 200
    state_space = env.observation_space.shape[0]
else: 
    model = CategoricalCnnDQN
    lr = 2e-4
    replay_buffer_size = 100000
    update_target_every = 1000
    env = wrap_pytorch(wrap_deepmind(env))
    state_space = env.observation_space.shape

args = to_attr(args_dict)

# setup loss function
if 'kl' in args.loss.lower(): 
    loss_fn = KL(args)
elif 'wasserstein' in args.loss.lower():
    loss_fn = Wasserstein(args)
elif 'cramer' in args.loss.lower():
    loss_fn = Cramer(args)

# initialize replay buffer
replay_buffer = ReplayBuffer(replay_buffer_size)
logger = Logger(args.base_dir)

# Logging
maybe_create_dir(args.base_dir)
print_and_save_args(args, args.base_dir)
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir, 'TB'))
print_and_save_args(args, args.base_dir)

# build models
current_model, target_model = [model(state_space, 
                                     env.action_space.n, 
                                     args.num_atoms, 
                                     args.Vmin, 
                                     args.Vmax) for _ in range(2)]

if USE_CUDA: 
    current_model, target_model = current_model.cuda(), target_model.cuda()

optimizer = optim.Adam(current_model.parameters(), lr)
update_target(current_model, target_model)

# logging placeholders
losses, episode_rewards = [], []
episode_reward, episode_iters, episodes = 0, 0, 0
state = env.reset()

# training loop
for i in range(1, num_frames + 1):
    action = current_model.act(state)
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward
    episode_iters  += 1

    if len(replay_buffer) >= args.batch_size and current_model.training: 
        states, actions, rewards, next_states, dones = replay_buffer.sample(args.batch_size, 
                                                                            cuda=USE_CUDA, 
                                                                            to_pytorch=True)  
        # calculate source distribution
        source_dist = current_model(states)
        actions = actions.unsqueeze(1).unsqueeze(1).expand(args.batch_size, 1, args.num_atoms)
        source_dist = source_dist.gather(1, actions).squeeze(1)       
 
        # calculate target distribution
        target_dist, bins = target_distribution(next_states, rewards, dones, target_model, args)
       
        loss = loss_fn(source_dist, bins if args.no_projection else target_dist)
        losses += [loss.data[0]]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        current_model.reset_noise()
        target_model.reset_noise()

    if i % update_target_every == 0:
        update_target(current_model, target_model)

    if done:
        print_and_log_scalar(writer, 'episode_loss', np.mean(losses), episodes)
        print_and_log_scalar(writer, 'episode_reward', episode_reward, episodes)
        print_and_log_scalar(writer, 'episode_iters', episode_iters, episodes)
        print('')

        state = env.reset()
        episode_reward = 0
        episode_iters = 0
        episodes += 1

        # plot(i, episode_rewards, losses, bin_size=5, save=True)

'''
Eval Mode
'''

eval_episode_rewards = []
episode_reward, episode_iters, episode = 0, 0, 0
state = env.reset()
current_model = current_model.eval()

while episode < args.eval_runs:
    action = current_model.act(state)
    state, reward, done, _ = env.step(action)
    episode_reward += reward
    episode_iters += 1

    if done:
        print_and_log_scalar(writer, 'eval_episode_reward', episode_reward, episodes)
        prind_and_log_scalar(writer, 'eval_episode_iters', episode_iters, episodes)
        state = env.reset()
        episode_reward = 0
        episode_iters = 0
