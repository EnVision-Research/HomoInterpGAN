'''
TrainingFramework

TrainingFrameworkv2: compared with the first version, the optimizer is an independent class of TrainingFramework


'''
from __future__ import print_function
from torch.utils.data import DataLoader
from .logger import logger
from tqdm import tqdm
import yaml
import os
log = logger(True)



class TrainEngine(object):
    def __init__(self, dataset, optimizer, batch_size, data_adapter=None, num_workers=8, recover_step_epoch=True):
        super(TrainEngine, self).__init__()
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                     drop_last=True)
        self.data_adapter = data_adapter
        self.optimizer = optimizer
        self.recover_step_epoch = recover_step_epoch
        with open(optimizer.opt.save_dir + '/options.txt', 'w') as f:
            print(optimizer.opt, file=f)
            print(optimizer.opt)

    def run(self, epoch):
        global_step = 0
        e0 = 0
        if self.recover_step_epoch:  # recover the global step and the epoch
            if os.path.exists(self.optimizer.opt.save_dir + '/step_epoch.yaml'):
                log('recover_step_epoch')
                with open(self.optimizer.opt.save_dir + '/step_epoch.yaml', 'r') as f:
                    global_step_epoch = yaml.load(f)
                    global_step = int(global_step_epoch['global_step'])
                    e0 = int(global_step_epoch['epoch'])
        for e in range(e0, epoch):
            for iter, data in enumerate(tqdm(self.dataloader)):
                if self.data_adapter is not None:
                    data = self.data_adapter(data)
                self.optimizer.set_input(data)
                # if iter == 0:
                #     self.optimizer.add_summary_heavy(global_step)
                self.optimizer.optimize_parameters(global_step)
                if global_step % 100 == 0:
                    tqdm.write(self.optimizer.print_current_errors(e, iter, record_file=self.optimizer.opt.save_dir,
                                                                   print_msg=False))
                    self.optimizer.add_summary(global_step)
                if (global_step) % 1000 == 0:
                    log('save samples ', global_step)
                    self.optimizer.save_samples(global_step)
                if global_step > 0 and global_step % 2000 == 0:
                    self.optimizer.save()
                global_step += 1
                with open(self.optimizer.opt.save_dir + '/step_epoch.yaml', 'w') as f:
                    global_step_epoch = {}
                    global_step_epoch['global_step'] = global_step
                    global_step_epoch['epoch'] = e
                    yaml.dump(global_step_epoch, f, default_flow_style=False)
            if e % 10 == 0 and e > 0:
                self.optimizer.save(e)
            # self.optimizer.save(e)
            self.optimizer.add_summary_heavy(e)
        self.optimizer.save()


