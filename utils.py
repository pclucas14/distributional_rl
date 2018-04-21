import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt


'''
returns the target distribution projected back on the valid support.
adapted from https://github.com/higgsfield/RL-Adventure/
'''

def target_distribution(next_state, rewards, dones, target_model, args):
    # make sure next_state has a batch axis
    # TODO : maximize gpu usage
    # next_state, rewards, dones = next_state.cpu(), rewards.cpu(), dones.cpu()
    
    if len(next_state.size()) == 1: 
        next_state = next_state.unsqueeze(0)

    batch_size  = next_state.size(0)
    
    delta_z = float(args.Vmax - args.Vmin) / (args.num_atoms - 1)
    support = torch.linspace(args.Vmin, args.Vmax, args.num_atoms)
    if args.cuda : support = support.cuda()

    # output of target_model is a **normalized** dist
    next_dist_raw  = target_model(next_state).data
    next_dist      = next_dist_raw * support
    next_action    = next_dist.sum(2).max(1)[1]
    next_action    = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
    next_dist      = next_dist.gather(1, next_action).squeeze(1)
    next_dist_raw  = next_dist_raw.gather(1, next_action).squeeze(1)
        
    if not isinstance(rewards, float):
        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones   = dones.unsqueeze(1).expand_as(next_dist)
    
    support = support.unsqueeze(0).expand_as(next_dist)
   
    Tz = rewards + (1 - dones) * args.gamma * support
    Tz = Tz.clamp(min=args.Vmin, max=args.Vmax)
    b  = (Tz - args.Vmin) / delta_z
    l  = b.floor().long() # lower bound of each bin
    u  = b.ceil().long()  # upper bound of each bin
        
    offset = torch.linspace(0, (args.batch_size - 1) * args.num_atoms, args.batch_size).long() \
                    .unsqueeze(1).expand(args.batch_size, args.num_atoms)

    proj_dist = torch.zeros(next_dist.size())  
    next_dist = next_dist_raw

    if args.cuda:
        offset = offset.cuda()
        proj_dist = proj_dist.cuda()
    
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float()) ).view(-1))
    
    # in the event that the upper bound equals the lower bound, make sure no mass vanishes
    next_dist_eq = u == l
    next_dist_eq = next_dist * next_dist_eq.float()
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist_eq).view(-1))
    
    return proj_dist


'''
Update the Target Model with weights from the current model
'''
def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

'''
convert a dictonnary to a argparse-like object
'''
def to_attr(args):
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    return AttrDict(args)

'''
plot training statistic
'''
def plot(frame_idx, rewards, losses, bin_size=5):
    plt.close()
    if len(rewards) > bin_size:
        rewards = np.array(rewards)
        remainder = rewards.shape[0] % bin_size
        rewards = rewards[remainder:] 
        rewards = rewards.reshape((-1, bin_size)).mean(axis=1)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    if len(losses) > bin_size:
        losses = np.array(losses)
        remainder = losses.shape[0] % bin_size
        losses = losses[remainder:] 
        losses = losses.reshape((-1, bin_size)).mean(axis=1)
    plt.plot(losses)
    plt.draw()
    plt.pause(0.001)
