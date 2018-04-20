import torch
import torch.nn as nn
from pyemd import * 
from torch.autograd import Variable
import numpy as np

'''
Abstract Class for Losses
'''
class Loss():
    def __init__(self, args):
        self.args = args

    '''
    input: numpy array, FloatTensor or Variable
    returns : a Variable
    '''
    def process_input_(self, dist, requires_grad):
        var = lambda x : torch.autograd.Variable(x, requires_grad=requires_grad)
        Variable = lambda x : var(x).cuda() if self.args.cuda else var(x)

        if isinstance(dist, np.ndarray):
            dist = Variable(torch.from_numpy(dist).float())
        elif isinstance(dist, torch.FloatTensor):
            dist = Variable(dist)
        elif isinstance(dist, torch.DoubleTensor):
            dist = Variable(dist).float()

        if not isinstance(dist, torch.autograd.Variable): 
            raise TypeError('input to loss should be numpy array, Tensor \
                or a Variable')

        # assert that the returned array has three dimensions
        if len(dist.size()) == 1: 
            dist = dist.unsqueeze(0)

        if len(dist.size()) != 2:
            raise ValueError('input should have 2 or 3 dimensions, not {} \
            '.format(len(dist.size())))
        
        return dist


    def calculate_loss_(self, source_dist, target_dist):
        raise NotImplementedError('Abstract method : should be implemented by subclasses')


    def __call__(self, source_dist, target_dist):
        source_dist = self.process_input_(source_dist, True)
        target_dist = self.process_input_(target_dist, False)

        return self.calculate_loss_(source_dist, target_dist)
        

class Wasserstein(Loss):
    def __init__(self, args):
        super(Wasserstein, self).__init__(args)
        print('using Wasserstein loss')

        distance_matrix = np.zeros((args.num_atoms, args.num_atoms))
        for i in range(args.num_atoms):
            for j in range(args.num_atoms):
                distance_matrix[i, j] = max(i - j, j - i)

        self.dist_matrix = distance_matrix
        self.dist_tensor = Variable(torch.from_numpy(distance_matrix)).float()
        self.dist_tensor = self.dist_tensor.unsqueeze(0)

    def get_transport_plans(self, source_dist, target_dist):
        all_plans = []
        for i in range(source_dist.size(0)):
            transport_plan = emd_with_flow(target_dist[i].data.double().numpy(), 
                                           source_dist[i].data.double().cpu().numpy(),
                                           self.dist_matrix)[1]
            transport_plan = Variable(torch.from_numpy(np.array(transport_plan)))
            all_plans += [transport_plan]
        
        out = torch.stack(all_plans, dim=0).float()
        return out.cuda() if source_dist.is_cuda else out
    
    def calculate_loss_(self, source_dist, target_dist):
        transport_plans = self.get_transport_plans(source_dist, target_dist)
        source_dist = source_dist.unsqueeze(1).repeat(1, self.args.num_atoms, 1)
        normalized_plan = transport_plans / source_dist.detach()
        cost = normalized_plan * self.dist_tensor * source_dist
        return cost.sum()


class KL(Loss):
    def __init__(self, args):
        super(KL, self).__init__(args)
        print('using KL loss')

    def calculate_loss_(self, source_dist, target_dist):
        source_dist.data.clamp(0.01, 0.99)
        return  - (target_dist * source_dist.log()).sum()

class Cramer(Loss):
    def __init__(self, args):
       print('using CRAMER loss')
       super(Cramer, self).__init__(args)
       mask = np.zeros((args.num_atoms, args.num_atoms))
       for i in range(args.num_atoms + 1):
           for j in range(i):
               mask[i-1, j] = 1
       
       self.mask = Variable(torch.from_numpy(mask).float()) 
   
    def get_cdf(self, categorical_dist):
        dist = categorical_dist.unsqueeze(1).expand(-1, self.args.num_atoms, -1)
        cdf = (dist * self.mask.unsqueeze(0)).sum(dim=-1)
        return cdf 

    def calculate_loss_(self, source_dist, target_dist):
        source_cdf = self.get_cdf(source_dist)
        target_cdf = self.get_cdf(target_dist)
        delta = source_cdf - target_cdf
        return (delta ** 2).sum() ** (.5)
