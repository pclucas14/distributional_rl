import os
import numpy as np

class Logger():
    def __init__(self, path):
        self.path = path
        # create appropriate directory
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            # check if job terminated successfully
            if os.path.isfile(os.path.join(path, 'terminated.txt')) and 'test' not in path:
                raise Exception('this job has already been run successfully') 
    
        self.episode_rewards = []   
        self.episode_losses  = []
        self.episode_steps   = []

        self.eval_episode_rewards = []   
        self.eval_episode_steps   = []

    def log_train_episode(self, reward, loss, step):
        self.episode_rewards += [reward]
        self.episode_losses  += [loss]
        self.episode_steps   += [step]

        assert len(self.episode_rewards) == len(self.episode_steps) == len(self.episode_losses)

    def log_eval_episode(self, reward, step):
        self.eval_episode_rewards += [reward]
        self.eval_episode_steps   += [step]

        assert len(self.eval_episode_rewards) == len(self.eval_episode_steps)  
    
    def save_run(self):
        fmt = '%10.5f'
        np.savetxt(os.path.join(self.path, 'train_episode_rewards.txt'), 
                   np.array(self.episode_rewards), fmt=fmt)
        np.savetxt(os.path.join(self.path, 'train_episode_losses.txt'), 
                   np.array(self.episode_losses), fmt=fmt)
        np.savetxt(os.path.join(self.path, 'train_episode_steps.txt'),
                   np.array(self.episode_steps), fmt=fmt)

        if len(self.eval_episode_rewards) > 0: 
            np.savetxt(os.path.join(self.path, 'eval_episode_rewards.txt'), 
                       np.array(self.eval_episode_rewards), fmt=fmt)
            np.savetxt(os.path.join(self.path, 'eval_episode_steps.txt'),
                   np.array(self.eval_episode_steps), fmt=fmt)

    def log_to_common_file(self, path_to_file):
        open_as = 'a' if os.path.isfile(path_to_file) else 'w'
        with open(path_to_file, open_as) as log:
            avg_reward = np.mean(self.eval_episode_rewards)
            log.write('{} SCORE {:10.5f}\n'.format(self.path, avg_reward))

    def signal_end(self):
        with open(os.path.join(self.path, 'terminated.txt'), 'w') as f:
            f.write('job terminated successfully')
