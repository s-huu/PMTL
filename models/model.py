"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import random
import copy
import torch
from datetime import datetime

from baseline_constants import ACCURACY_KEY, OptimLoggingKeys, AGGR_MEAN, AGGR_MEDIAN, AGGR_KRUM, AGGR_MULTIKRUM

from utils.model_utils import batch_data




class Model(ABC):

    def __init__(self, lr, seed, max_batch_size, optimizer=None):
        self.lr = lr
        self.optimizer = optimizer
        self.rng = random.Random(seed)
        self.size = None


        # largest batch size for which GPU will not run out of memory
        self.max_batch_size = max_batch_size if max_batch_size is not None else 2 ** 14
        print('***** using a max batch size of', self.max_batch_size)

        self.flops = 0

    def train(self, data, num_epochs=1, batch_size=100, lr=None, malicious=False, q=0, qffl=False, attack="label_poison", lmbda=0.5, ewc=False, gamma=0.5):
        # print(self.optimizer.personalized)
        if lr is None:
            lr = self.lr
        averaged_loss = 0.0
        batched_x, batched_y = batch_data(data, batch_size=100, rng=self.rng, shuffle=True, malicious=malicious)
        if self.optimizer.w is None:
            self.optimizer.initialize_w()

        # print(self.optimizer.global_w_on_last_update)
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&& training one client")
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_of_samples = 0
            if ewc:
                weights = copy.deepcopy(self.optimizer.model.trainable_parameters())
                weights = [torch.zeros_like(layer) for layer in weights]
                for _ in range(2):
                    for i, (raw_x_batch,raw_y_batch) in enumerate(zip(batched_x, batched_y)):
                        input_data = self.process_x(raw_x_batch)
                        target_data = self.process_y(raw_y_batch)
                        weights = self.optimizer.ewc_weights(input_data, target_data, weights)
                for w in weights:
                    w /= 2

            for i, (raw_x_batch,raw_y_batch) in enumerate(zip(batched_x, batched_y)):
                input_data = self.process_x(raw_x_batch)
                target_data = self.process_y(raw_y_batch)
                num_of_samples += len(target_data)
                if ewc:
                    loss = self.optimizer.run_step_ewc(input_data, target_data, lmbda, weights)
                else:
                    loss = self.optimizer.run_step(input_data, target_data, lmbda)
                total_loss += loss
            averaged_loss = total_loss / len(batched_x)
        self.optimizer.end_local_updates() # required for pytorch models
        if malicious:
            if self.optimizer.personalized:
                if attack == "random":
                    update = np.random.randn(*self.optimizer.global_w.shape)
                elif attack == "model_replacement":
                    update = num_of_samples*np.copy(self.optimizer.global_w - self.optimizer.global_w_on_last_update)
                elif attack == "label_poison":
                    update = np.copy(self.optimizer.global_w - self.optimizer.global_w_on_last_update)
            else:
                if attack == "random":
                    update = np.random.randn(*self.optimizer.w.shape)
                elif attack == "model_replacement":
                    update = num_of_samples*np.copy(self.optimizer.w - self.optimizer.w_on_last_update)
                elif attack == "label_poison":
                    update = np.copy(self.optimizer.w - self.optimizer.w_on_last_update)
        else:
            if self.optimizer.personalized:
                norm = np.linalg.norm(self.optimizer.global_w - self.optimizer.global_w_on_last_update)
                update = np.copy((self.optimizer.global_w - self.optimizer.global_w_on_last_update) * min(1, gamma/norm))
            else:
                norm = np.linalg.norm(self.optimizer.w - self.optimizer.w_on_last_update)
                update = np.copy((self.optimizer.w - self.optimizer.w_on_last_update) * min(1, gamma/norm))
        

        self.optimizer.update_w()

        comp = num_epochs * len(batched_y) * batch_size * self.flops
        # now = datetime.now()
        # print("training....", now)
        
        return comp, update, averaged_loss
    
    

    def test(self, eval_data, train_data=None, train_users=True, malicious=False):
        

        output = {'eval': [-float('inf'), -float('inf')], 'train': [-float('inf'), -float('inf')]}

        if self.optimizer.w is None:
            self.optimizer.initialize_w()

        total_loss, total_correct, count = 0.0, 0, 0
        batched_x, batched_y = batch_data(eval_data, self.max_batch_size, shuffle=False, eval_mode=True, malicious=malicious)
        for i,(x, y) in enumerate(zip(batched_x, batched_y)):
            x_vecs = self.process_x(x)
            labels = self.process_y(y)
            # print(x_vecs.shape)
            loss = self.optimizer.loss(x_vecs, labels, eval_mode=True)
            # loss = 1
            correct = self.optimizer.correct(x_vecs, labels)
            

            total_loss += loss * len(y)  # loss returns average over batch
            total_correct += correct  # eval_op returns sum over batch
            count += len(y)
        loss = total_loss / count
        acc = total_correct / count
        if train_users:
            output['train'] = [loss, acc]
        else:
            output['eval'] = [loss, acc]
        
        try:
            if self.optimizer.w_init is None:
                self.optimizer.w_init = self.optimizer.w
        except:
            self.optimizer.w_init = self.optimizer.w
        norm_diff = np.linalg.norm(self.optimizer.w-self.optimizer.w_init)
        # norm_diff = 0

        # now = datetime.now()
        # print("testing....", now)

        return {
                ACCURACY_KEY: output['eval'][1],
                OptimLoggingKeys.TRAIN_LOSS_KEY: output['train'][0],
                OptimLoggingKeys.TRAIN_ACCURACY_KEY: output['train'][1],
                OptimLoggingKeys.EVAL_LOSS_KEY: output['eval'][0],
                OptimLoggingKeys.EVAL_ACCURACY_KEY: output['eval'][1],
                OptimLoggingKeys.DIFF_NORM: norm_diff
                }



    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        return np.asarray(raw_x_batch)


    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        return np.asarray(raw_y_batch)


class ServerModel:
    def __init__(self, model):
        self.model = model
        self.rng = model.rng

    @property
    def size(self):
        return self.model.optimizer.size()

    @property
    def cur_model(self):
        return self.model

    def send_to(self, clients):
        """Copies server model variables to each of the given clients

        Args:
            clients: list of Client objects
        """
        var_vals = {}
        for c in clients:
            c.model.optimizer.reset_w(self.model.optimizer.w)
            c.model.size = self.model.optimizer.size()


    @staticmethod
    def weighted_average_oracle(points, weights, k_aggregator=False, k_aggr_ratio=1, losses=[]):
        """Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """ 
            
        tot_weights = np.sum(weights)

        weighted_updates = np.zeros_like(points[0])

        for w, p in zip(weights, points):
            weighted_updates += (w / tot_weights) * p

        return weighted_updates
  

    def update(self, updates, aggregation=AGGR_MEAN, clipping=False, k_aggregator=False, k_aggr_ratio=1, losses=[], sigma=5e-2):
        """Updates server model using given client updates.

        Args:
            updates: list of (num_samples, update), where num_samples is the
                number of training samples corresponding to the update, and update
                is a list of variable weights
            aggregation: Algorithm used for aggregation. Allowed values are:
                [ 'mean'], i.e., only support aggregation with weighted mean
        """
        if len(updates) == 0:
            print('No updates obtained. Continuing without update')
            return 1, False
        def accept_update(u):
            # norm = np.linalg.norm([np.linalg.norm(x) for x in u[1]])
            norm = np.linalg.norm(u[1])
            return not (np.isinf(norm) or np.isnan(norm))
        all_updates = updates
        updates = [u for u in updates if accept_update(u)]
        if len(updates) < len(all_updates):
            print('Rejected {} individual updates because of NaN or Inf'.format(len(all_updates) - len(updates)))
        if len(updates) == 0:
            print('All individual updates rejected. Continuing without update')
            return 1, False

        points = [u[1] for u in updates]
        alphas = [u[0] for u in updates]
        if clipping:
            clipping_threshold = np.median([np.linalg.norm(p) for p in points])
            points = [self.L2_clip(p,clipping_threshold) for p in points]
        if aggregation == AGGR_MEAN:
            weighted_updates = self.weighted_average_oracle(points, alphas, k_aggregator, k_aggr_ratio, losses)
            num_comm_rounds = 1
        elif aggregation == AGGR_MEDIAN:
            weighted_updates = np.median(points)
            num_comm_rounds = 1
        elif aggregation == AGGR_KRUM:
            weighted_updates = points[self.krum_func(points,5)]
            num_comm_rounds = 1
        elif aggregation == AGGR_MULTIKRUM:
            score_indices = self.multi_krum_func(points,5,5)
            weighted_updates = self.weighted_average_oracle(points[score_indices], alphas[score_indices], k_aggregator, k_aggr_ratio, losses)
            num_comm_rounds = 1
        else:
            raise ValueError('Unknown aggregation strategy: {}'.format(aggregation))

        # update_norm = np.linalg.norm([np.linalg.norm(v) for v in weighted_updates])
        update_norm = np.linalg.norm(weighted_updates)
        print(update_norm)

        self.model.optimizer.w += np.array(weighted_updates)+np.random.randn(weighted_updates.shape[0])*sigma
        self.model.optimizer.reset_w(self.model.optimizer.w)  # update server model
        updated = True

        return num_comm_rounds, updated


class Optimizer(ABC):

    def __init__(self, starting_w=None, loss=None, loss_prime=None):
        self.w = starting_w
        self.w_on_last_update = np.copy(starting_w)
        self.optimizer_model = None

    @abstractmethod
    def loss(self, x, y):
        return None

    @abstractmethod
    def gradient(self, x, y):
        return None

    @abstractmethod
    def run_step(self, batched_x, batched_y): # should run a first order method step and return loss obtained
        return None

    @abstractmethod
    def correct(self, x, y):
        return None

    def end_local_updates(self):
        pass

    def reset_w(self, w):
        self. w = np.copy(w)
        self.w_on_last_update = np.copy(w)

