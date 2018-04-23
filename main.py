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
parser.add_argument('--exp_path', type=str, default='experiments/test')
parser.add_argument('--exp_common_log', type=str, default='experiments/common.txt')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

# reproducibility
if args.seed:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

args_dict = vars(args)

# GPU settings 
USE_CUDA = torch.cuda.is_available()
args_dict['cuda'] = USE_CUDA

# add environment settings
if 'cartpole' in args.env.lower():
    env = gym.make('CartPole-v0')
    args_dict['Vmin'] = 0
    args_dict['Vmax'] = 200
    num_frames = 15000
elif 'acrobot' in args.env.lower():
    env = gym.make('Acrobot-v1')
    args_dict['Vmin'] = -500
    args_dict['Vmax'] = 0
    num_frames = 25000
elif 'pong' in args.env.lower():
    env = make_atari('PongNoFrameskip-v4')
    args_dict['Vmin'] = -20
    args_dict['Vmax'] = 20
    num_frames = 1000000
else: 
    raise ValueError('invalid environment')

# more general settings
if args.env.lower() in ['cartpole', 'acrobot']: 
    model = CategoricalDQN
    lr = 8e-4
    replay_buffer_size = 10000
    update_target_every = 100
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
logger = Logger(args.exp_path)

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
        target_dist = target_distribution(next_states, rewards, dones, target_model, args)
       
        loss = loss_fn(source_dist, target_dist)
        losses += [loss.data[0]]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        current_model.reset_noise()
        target_model.reset_noise()

    if i % update_target_every == 0:
        update_target(current_model, target_model)

    if done:
        episode_loss = losses[-episode_iters:] 
        if len(losses) > 0: logger.log_train_episode(episode_reward, np.mean(episode_loss), episode_iters)
        state = env.reset()
        episode_rewards += [episode_reward]
        episode_reward = 0
        episode_iters = 0
        episodes += 1

        if episodes % 5 == 0 and args.verbose: 
            print('last 5 episode rewards {}'.format(str(episode_rewards[-5:])))
            print('last 5 losses {}'.format(str(losses[-5:])))
            # plot(i, episode_rewards, losses, bin_size=5)

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
        logger.log_eval_episode(episode_reward, episode_iters)
        state = env.reset()
        eval_episode_rewards += [episode_reward]
        episode_reward = 0
        episode_iters = 0
        episode += 1

print('average eval reward : {}'.format(np.mean(eval_episode_rewards)))

# save logs 
logger.save_run()
logger.log_to_common_file(args.exp_common_log)
logger.signal_end()

# save the model
torch.save(current_model.state_dict(), os.path.join(args.exp_path, 'model.pth'))
