# -*- coding: utf-8 -*-
"""
Wrapper class for crop yield regression model employing a ResNet50 or densenet in PyTorch in a transfer learning approach using
RGB remote sensing observations
"""

# Built-in/Generic Imports

# Libs
import math
import time, gc, os
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torchvision.models as models
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from torchvision.models.densenet import DenseNet
from torchvision.models.resnet import ResNet

# Own modules
from BaselineModel import BaselineModel


__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'

class RGBYieldRegressor_Trainer:
    def __init__(self, device, workers, pretrained:bool = True, tune_fc_only:bool = True, architecture: str = 'resnet18', criterion=None):
        # self.k = k
        self.architecture = architecture
        self.pretrained = pretrained
        self.lr = None
        self.momentum = 0.9
        self.wd = None
        self.batch_size = None
        # self.training_response_standardizer = None

        self.device = device
        self.workers = workers

        num_target_classes = 1
        if self.architecture == 'resnet18':
            # init resnet18 with FC exchanged
            if pretrained:
                self.model = models.resnet18(weights='DEFAULT')
            else:
                self.model = models.resnet18(weights=None)
            num_filters = self.model.fc.in_features
            self.num_filters = num_filters
            self.model.fc = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(num_filters, num_target_classes))

        elif self.architecture == 'densenet':
            # init densenet with FC exchanged
            self.model = models.densenet121(pretrained=pretrained)
            num_filters = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(num_filters, num_target_classes)
            )
        elif self.architecture == 'short_densenet':
            # init custom DenseNet with one DenseBlock only
            self.model = DenseNet(growth_rate=32,
                                  block_config = (6,), # 12 , 24, 16)# initialize woth only 1 DenseBlock
                                  num_init_features=64,
                                  )
            # load pretrained version
            pretrained_densenet121 = models.densenet121(pretrained=pretrained)
            # copy weights
            pretrained_state_dict = self.model.state_dict()
            for name, param in pretrained_densenet121.state_dict().items():
                if name not in self.model.state_dict(): # copy only the first parts that are the same, skip rest
                    continue
                if ('norm5' in name) or ('classifier' in name): # skip last parts due to demand for retraining
                    continue
                pretrained_state_dict[name] = param
            self.model.load_state_dict(pretrained_state_dict)

            num_filters = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(num_filters, num_target_classes)
            )
        elif self.architecture == 'baselinemodel':
            self.model = BaselineModel()
            for child in list(self.model.children()):
                for param in child.parameters():
                    param.requires_grad = True

        # disable gradient computation for conv layers
        if pretrained and tune_fc_only:
            self.disable_all_but_fc_grads()

        # self.set_optimizer(scheduler=True)
        #
        if criterion is None:
            self.set_criterion()
        else:
            self.set_criterion(criterion=criterion)

    def print_children_require_grads(self):
        children = [child for child in self.model.children()]
        i = 0
        for child in children:  # freeze all layers up until self.model.fc
            params = child.parameters()
            for param in params:
                print('child_{} grad required: {}'.format(i,param.requires_grad))
            i+=1

    def enable_grads(self):
        # for child in list(self.model.children()):
        #     for param in child.parameters():
        #         param.requires_grad = True
        if isinstance(self.model, BaselineModel):
            self.model.blocks['block_0'].requires_grad_(True)
            self.model.blocks['block_1'].requires_grad_(True)
            self.model.blocks['block_2'].requires_grad_(True)
            self.model.blocks['block_3'].requires_grad_(True)
            self.model.blocks['block_4'].requires_grad_(True)
            self.model.blocks['block_5'].requires_grad_(True)
            self.model.blocks['block_6'].requires_grad_(True)
        elif isinstance(self.model, DenseNet):
            self.model.features.requires_grad_(True)
        elif isinstance(self.model, ResNet):
            for child in self.model.children():  # unfreeze all layers
                for param in child.parameters():
                    param.requires_grad = True

    # def save_SSL_fc_weights(self):
    #     if isinstance(self.model, BaselineModel):
    #         self.fc_7_weight = self.model.regressor.fc_7.weight
    #         self.fc_8_weight = self.model.regressor.fc_8.weight
    #         self.fc_9_weight = self.model.regressor.fc_9.weight
    #     else:
    #         raise NotImplementedError

    def reinitialize_fc_layers(self):
        if isinstance(self.model, BaselineModel):
            self.regressor = nn.Sequential(OrderedDict([
                ('fc_7', nn.Linear(in_features=5 * 5 * 128, out_features=1024)),
                ('relu_7', nn.ReLU(inplace=True)),
                ('fc_8', nn.Linear(in_features=1024, out_features=1))
            ]))
        elif isinstance(self.model, ResNet):
            self.model.fc = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(self.num_filters, 1))
        elif isinstance(self.model, DenseNet):
            self.model.classifier = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(self.num_filters, 1)
            )
        else:
            raise NotImplementedError

    def disable_all_but_fc_grads(self):
        if isinstance(self.model, BaselineModel):
            self.model.blocks['block_0'].requires_grad_(False)
            self.model.blocks['block_1'].requires_grad_(False)
            self.model.blocks['block_2'].requires_grad_(False)
            self.model.blocks['block_3'].requires_grad_(False)
            self.model.blocks['block_4'].requires_grad_(False)
            self.model.blocks['block_5'].requires_grad_(False)
            self.model.blocks['block_6'].requires_grad_(False)
        elif isinstance(self.model, DenseNet):
            self.model.features.requires_grad_(False)
        elif isinstance(self.model, ResNet):
            children = [child for child in self.model.children()]
            for child in children[:-1]:  # freeze all layers up until self.model.fc
                for param in child.parameters():
                    param.requires_grad = False
            for child in children[-1:]:  # enable gradient computation for self.model.fc
                for param in child.parameters():
                    param.requires_grad = True
        else:
            raise NotImplementedError

    def set_dataloaders(self, dataloaders):
        self.dataloaders = dataloaders

    # def set_training_response_standardizer(self, training_response_standardizer):
    #     self.training_response_standardizer = training_response_standardizer

    def set_hyper_parameters(self, lr, wd, batch_size, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.wd = wd
        self.batch_size = batch_size

    def set_optimizer(self, scheduler = True):
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                         lr=self.lr,
                                         momentum=self.momentum,
                                         weight_decay=self.wd)

        if scheduler == True:
            # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
            #                                                    base_lr=self.lr/1000,
            #                                                    max_lr=self.lr,
            #                                                    step_size_up=2000,
            #                                                    )
            iterations = math.floor(4209/self.batch_size)
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
                                                               base_lr=self.lr / 4,
                                                               max_lr=self.lr,
                                                               step_size_up=iterations*2,
                                                               )
        else:
            self.scheduler = None

    # def set_criterion(self, criterion=nn.MSELoss(reduction='mean')):
    def set_criterion(self, criterion=nn.L1Loss(reduction='mean')):
        self.criterion = criterion

    def start_timer(self, device=None):
        global start_time
        gc.collect()
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated(device=device)
            torch.cuda.synchronize()
        self.start_time = time.time()

    def end_timer_and_get_time(self, local_msg=''):
        if self.device == torch.device("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()
        print("\n" + local_msg)
        print("Total execution time = {} sec".format(end_time - self.start_time))
        print('Max memory used by tensors = {} bytes'.format(torch.cuda.max_memory_allocated()))
        return end_time - self.start_time

    def train_step(self):
        '''
        1 epoch training
        '''
        phase = 'train'
        running_loss = 0.0
        epoch_loss = 0.0

        scaler = GradScaler()
        batch_size = self.dataloaders['train'].batch_size
        if batch_size == 4:
            gradient_accumulations = 16 # accumulate for 4*16=64 batches -> update gradients every 64 mini batches
        elif batch_size == 8:
            gradient_accumulations = 8
        elif batch_size == 16:
            gradient_accumulations = 4
        elif batch_size == 32:
            gradient_accumulations = 2
        elif batch_size >= 64:
            gradient_accumulations = 1

        # Set model to training mode
        self.model.train()

        # Iterate over data.
        for batch_idx, batch in enumerate(self.dataloaders[phase]):

            # # zero the parameter gradients for every mini batch
            # self.optimizer.zero_grad()
            # # forward

            # with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.set_grad_enabled(phase == 'train'):
                with autocast():
                    # make prediction
                    outputs = self.model(inputs)
                    # print('Outside: input size', inputs.size(),
                    #       'output size', outputs.size())
                    # compute loss
                    loss = self.criterion(torch.flatten(outputs), labels.data)

                    scaler.scale(loss / gradient_accumulations).backward()
                    # # accumulate gradients
                    # loss.backward()

                    # update only for accumulated gradients of at least 64 batches
                    if (batch_idx + 1) % gradient_accumulations == 0:
                        scaler.step(self.optimizer) #invokes optimizer.step() if no inf/NaN found
                        scaler.update()
                        if self.scheduler is not None:
                            self.scheduler.step()
                        self.model.zero_grad()

                    # # update parameters
                    # self.optimizer.step()
                    # if self.scheduler is not None:
                    #     self.scheduler.step()
            # statistics
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        return epoch_loss

    def test(self, phase:str = 'test'):
        '''
        1 iteration testing on data set (default: test set)
        '''
        running_loss = 0.0
        epoch_loss = 0.0

        self.model.to(self.device)
        self.model.eval()  # Set model to evaluate mode
        # Iterate over data.
        for inputs, labels in self.dataloaders[phase]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                # make prediction
                outputs = self.model(inputs)
                # compute loss
                # if self.training_response_standardizer is not None:
                #     loss = self.criterion((torch.flatten(outputs)*self.training_response_standardizer['std'])+self.training_response_standardizer['mean'], labels.data)
                # else:
                #     loss = self.criterion(torch.flatten(outputs), labels.data)
                loss = self.criterion(torch.flatten(outputs), labels.data)
            # statistics
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        return epoch_loss

    def predict(self, phase:str = 'test'):
        '''
        prediction on dataloader[phase: str] (default: test set)
        '''
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluate mode

        # for each fold store labels and predictions
        local_preds = []
        local_labels = []
        # Iterate over data.
        for inputs, labels in self.dataloaders[phase]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                # make prediction
                # if self.training_response_standardizer is not None:
                #     y_hat = (torch.flatten(self.model(inputs))*self.training_response_standardizer['std'])+self.training_response_standardizer['mean']
                #     labels = (labels *self.training_response_standardizer['std'])+self.training_response_standardizer['mean']
                # else:
                y_hat = torch.flatten(self.model(inputs))
            local_preds.extend(y_hat.detach().cpu().numpy())
            local_labels.extend(labels.detach().cpu().numpy())

        return local_preds, local_labels

    def train(self, patience:int = 5, min_delta:float = 0.01, num_epochs:int = 200, min_epochs: int = 200):
        '''
        train model for epochs
        :return:
        '''
        since = time.time()
        test_mse_history = []
        train_mse_history = []

        best_model_wts = deepcopy(self.model.state_dict())
        best_loss = np.inf
        best_epoch = 0.0
        init_patience = patience

        self.model.to(self.device)
        self.model.zero_grad() # zero model gradient across all optimizer
        for epoch in range(num_epochs):
            if patience > 0:
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))
                print('-' * 10)
                for phase in ['train', 'val']:
                    if phase == 'train':
                        epoch_loss = self.train_step()
                        train_mse_history.append(epoch_loss)
                    if phase == 'val':
                        epoch_loss = self.test(phase='val')
                        test_mse_history.append(epoch_loss)

                        ## first check early stopping
                        # save model with tighter fit
                        # requires improvement in loss (i.e. smaller in loss) to be greater than min_delta
                        # if (epoch_loss - best_loss) < -1 * min_delta:
                        # if (best_loss - epoch_loss) / best_loss > min_delta:
                        # print("early stopping check: improvement = {} / {}; found = {}".format((best_loss - epoch_loss) / best_loss , min_delta, ((best_loss - epoch_loss) / best_loss) > min_delta))
                        if ((best_loss - epoch_loss) / best_loss) <= min_delta:
                            # if (best_loss - epoch_loss) > min_delta:
                            if epoch >= min_epochs:
                                patience -= 1
                                print('no sufficient improvement found, ticking patience {}/{}'.format(patience, init_patience))
                        else:
                            patience = init_patience
                        # ## second check best model
                        # if epoch_loss < best_loss:
                            print('saving best model')
                            best_loss = epoch_loss
                            best_model_wts = deepcopy(self.model.state_dict())
                            best_epoch = epoch


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # print('Best val MSE: {:4f}'.format(best_mse))
        print('Best val MSE: {:4f}'.format(best_loss))

        self.optimizer_state_dict = self.optimizer.state_dict()
        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.test_mse_history = test_mse_history
        self.train_mse_history = train_mse_history
        self.best_epoch = best_epoch

    def load_model_if_exists(self, model_dir=None, strategy=None, k=None):
        '''
        load self.model if exists and (data) parallelize if self.workers > 1
        :param model_dir: directory in which the model is located
        :param strategy: pre-training strategy to build the model name
        :param k: cross validation fold, i.e. model index
        :return: True if model successfully loaded
        '''
        model_loaded = False
        if strategy is not None:
            checkpoint_path = os.path.join(model_dir, 'model_f' + str(k) + '_' + strategy + '.ckpt')
        else:
            checkpoint_path = os.path.join(model_dir, 'model_f' + str(k) + '.ckpt')
        if os.path.exists(checkpoint_path):
            print('\tLoading trained model: {}'.format(checkpoint_path))
            # load trained model weights with respect to possible parallelization and device
            if torch.cuda.is_available():
                state_dict = torch.load(checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            if self.workers > 1:
                self.model = nn.DataParallel(self.model)

            self.model.load_state_dict(state_dict)
            model_loaded = True
        return model_loaded

    def parallize_and_to_device(self):
        '''
        data parallelize self.model if self.workers>1
        :return: None
        '''
        if self.workers > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

