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

from lightly.models.modules import SimCLRProjectionHead, SimSiamPredictionHead, SimSiamProjectionHead, BarlowTwinsProjectionHead #, VICRegProjectionHead
from lightly.models.modules.heads import VicRegLLocalProjectionHead

from lars import LARS

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


class VICReg(nn.Module):
    def __init__(self, backbone, num_filters, prediction_head:bool=False):
        super().__init__()
        self.backbone = backbone
        if not prediction_head:
            self.projection_head = BarlowTwinsProjectionHead(num_filters, 2048, 2048)
        else:
            self.projection_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(num_filters, 1))

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

class VICRegL(nn.Module):
    def __init__(self, backbone, num_filters, prediction_head:bool=False):
        super().__init__()
        self.backbone = backbone
        self.SSL_training = not prediction_head

        if not prediction_head:
            self.projection_head = BarlowTwinsProjectionHead(num_filters, 2048, 2048)
            self.local_projection_head = VicRegLLocalProjectionHead(num_filters, 128, 128)
            self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        else:
            self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.projection_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(num_filters, 1))

    def forward(self, x):
        if self.SSL_training:
            x = self.backbone(x)
            y = self.average_pool(x).flatten(start_dim=1)
            z = self.projection_head(y)
            y_local = x.permute(0, 2, 3, 1)  # (B, D, W, H) to (B, W, H, D)
            z_local = self.local_projection_head(y_local)
            return z, z_local
        else:
            x = self.backbone(x)
            # x = self.backbone(x).flatten(start_dim=1)
            # print("DEBUG")
            # print(x.size())
            # print(self.projection_head)
            x = self.average_pool(x)
            x = torch.flatten(x,1)
            z = self.projection_head(x)
            return z

class SimSiam(nn.Module):
    def __init__(self, backbone, num_filters, prediction_head:bool=False):
        super().__init__()
        self.backbone = backbone
        self.SSL_training = not prediction_head

        # dimension of the embeddings
        # num_ftrs = 512
        # dimension of the output of the prediction and projection heads
        out_dim = proj_hidden_dim = 512
        # the prediction head uses a bottleneck architecture
        # pred_hidden_dim = 128
        pred_hidden_dim = 512

        if not prediction_head:
            self.projection_head = SimSiamProjectionHead(num_filters, proj_hidden_dim, out_dim)
            self.prediction_head = SimSiamPredictionHead(out_dim, pred_hidden_dim, out_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(num_filters, 1))
            self.prediction_head = None

    def forward(self, x):
        if self.SSL_training:
            f = self.backbone(x).flatten(start_dim=1)
            z = self.projection_head(f)
            p = self.prediction_head(z)
            z = z.detach()
            return z, p
        else:
            x = self.backbone(x).flatten(start_dim=1)
            z = self.projection_head(x)
            return z

class SimCLR(nn.Module):
    def __init__(self, backbone, num_filters, prediction_head:bool=False):
        super().__init__()
        self.backbone = backbone
        if not prediction_head:
            self.projection_head = SimCLRProjectionHead(512, 512, 128)
        else:
            self.projection_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(num_filters, 1))

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

class RGBYieldRegressor_Trainer:
    def __init__(self, device, workers, pretrained:bool = True, tune_fc_only:bool = True, architecture: str = 'resnet18', criterion=None, SSL=None, prediction_head:bool=False):
        # self.k = k
        self.architecture = architecture
        self.pretrained = pretrained
        self.lr = None
        self.momentum = 0.9
        self.wd = None
        self.batch_size = None
        self.SSL = SSL
        self.prediction_head=prediction_head
        # self.training_response_standardizer = None

        self.device = device
        self.workers = workers

        self.lrs = []

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

        elif self.architecture == 'SimCLR':
            # init resnet18 with FC exchanged
            if pretrained:
                raise 'pretrained SlimCLR not implemented'
            else:
                resnet = models.resnet18(weights=None)
                self.num_filters = resnet.fc.in_features
            backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.model = SimCLR(backbone, num_filters=self.num_filters, prediction_head=prediction_head)

        elif self.architecture == 'VICReg':
            # init resnet18 with FC exchanged
            if pretrained:
                raise 'pretrained VICReg not implemented'
            else:
                resnet = models.resnet18(weights=None)
                # resnet = models.resnet50(weights=None)
                self.num_filters = resnet.fc.in_features
            backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.model = VICReg(backbone, num_filters=self.num_filters, prediction_head=prediction_head)

        elif self.architecture == 'VICRegL':
            # init resnet18 with FC exchanged
            if pretrained:
                raise 'pretrained VICRegL not implemented'
            else:
                resnet = models.resnet18(weights=None)
                self.num_filters = resnet.fc.in_features
            backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.model = VICRegL(backbone, num_filters=self.num_filters, prediction_head=prediction_head)

        elif self.architecture == 'VICRegLConvNext':
            # init resnet18 with FC exchanged
            if pretrained:
                raise 'pretrained VICRegL not implemented'
            else:
                resnet = models.convnext_small(weights=None)
                self.num_filters = 768 #resnet.fc.in_features
            backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.model = VICRegL(backbone, num_filters=self.num_filters, prediction_head=prediction_head)
        elif self.architecture == 'SimSiam':
            # init resnet18 with FC exchanged
            if pretrained:
                raise 'pretrained SimSiam not implemented'
            else:
                resnet = models.resnet18(weights=None)
                # resnet = models.resnet50(weights=None)
                self.num_filters = resnet.fc.in_features
            backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.model = SimSiam(backbone, num_filters=self.num_filters, prediction_head=prediction_head)

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
        elif isinstance(self.model, ResNet) or isinstance(self.model, SimCLR) or isinstance(self.model, VICReg) or isinstance(self.model, SimSiam) or isinstance(self.model, VICRegL):
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
        elif isinstance(self.model, SimCLR):
            if self.SSL is None:
                self.model.projection_head = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Linear(self.num_filters, 1))
            elif self.SSL=='SimCLR':
                self.model.projection_head = SimCLRProjectionHead(512, 512, 128)
        elif isinstance(self.model, VICReg):
            if self.SSL is None:
                self.model.projection_head = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Linear(self.num_filters, 1))
            elif self.SSL=='VICReg':
                self.model.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)
        elif isinstance(self.model, VICRegL):
            if self.SSL is None:
                self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.model.projection_head = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Linear(self.num_filters, 1))
            elif self.SSL=='VICRegL' or 'VICRegLConvNext':
                self.model.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)
                self.model.local_projection_head = VicRegLLocalProjectionHead(512, 128, 128)
                self.model.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif isinstance(self.model, SimSiam):
            if self.SSL is None:
                self.model.projection_head = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Linear(self.num_filters, 1))
                self.model.prediction_head = None
            elif self.SSL=='SimSiam':
                self.projection_head = SimSiamProjectionHead(512, 512, 128)
                self.prediction_head = SimSiamPredictionHead(128, 64, 128)

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
        elif isinstance(self.model, ResNet) or isinstance(self.model, SimCLR) or isinstance(self.model, VICReg) or isinstance(self.model, SimSiam):
            children = [child for child in self.model.children()]
            for child in children[:-1]:  # freeze all layers up until self.model.fc
                for param in child.parameters():
                    param.requires_grad = False
            for child in children[-1:]:  # enable gradient computation for self.model.fc
                for param in child.parameters():
                    param.requires_grad = True
        elif isinstance(self.model, VICRegL):
            # disable backbone
            backbone_children = [child for child in self.model.backbone.children()]
            for child in backbone_children:  # freeze all layers up until self.model.fc
                for param in child.parameters():
                    param.requires_grad = False
            # enable new projection_head
            ph_children = [child for child in self.model.projection_head.children()]
            for child in ph_children:
                for param in child.parameters():
                    param.requires_grad = True
            # disable local ph
            lph_children = [child for child in self.model.local_projection_head.children()]
            for child in lph_children:  # freeze all layers up until self.model.fc
                for param in child.parameters():
                    param.requires_grad = False
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

        # if self.SSL == 'VICRegL' or self.SSL == 'VICReg':
        #     self.optimizer = LARS(params=self.model.parameters(),
        #                                      lr=self.lr,
        #                                      momentum=self.momentum,
        #                                      weight_decay=self.wd,)
        #
        # else:
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
            # if self.SSL == 'SimCLR' or self.SSL == 'VICReg' or self.SSL == 'SimSiam' or self.SSL == 'VICRegL':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                        T_0=50,
                                                                        T_mult=2,
                                                                                  )
            # else:
            #     iterations = math.floor(4209/self.batch_size)
            #     self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
            #                                                        base_lr=self.lr / 4,
            #                                                        max_lr=self.lr,
            #                                                        step_size_up=iterations*2,
            #                                                        )
        else:
            self.scheduler = None
        print('using optimizer: {}; with scheduler: {}'.format(self.optimizer, self.scheduler))

    # def set_optimizer(self, scheduler = True):
    #
    #     # if self.SSL == 'VICRegL' or self.SSL == 'VICReg':
    #     #     self.optimizer = LARS(params=self.model.parameters(),
    #     #                                      lr=self.lr,
    #     #                                      momentum=self.momentum,
    #     #                                      weight_decay=self.wd,)
    #     #
    #     # else:
    #     self.optimizer = torch.optim.SGD(params=self.model.parameters(),
    #                                      lr=self.lr,
    #                                      momentum=self.momentum,
    #                                      weight_decay=self.wd)
    #
    #     if scheduler == True:
    #         self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
    #                                                            base_lr=self.lr/1000,
    #                                                            max_lr=self.lr,
    #                                                            step_size_up=2000,
    #                                                            )
    #         if self.SSL == 'SimCLR' or self.SSL == 'VICReg' or self.SSL == 'SimSiam' or self.SSL == 'VICRegL':
    #             self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
    #                                                                         T_0=50,
    #                                                                         T_mult=2,
    #                                                                                   )
    #         else:
    #             iterations = math.floor(4209/self.batch_size)
    #             self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
    #                                                                base_lr=self.lr / 4,
    #                                                                max_lr=self.lr,
    #                                                                step_size_up=iterations*2,
    #                                                                )
    #     else:
    #         self.scheduler = None
    #     print('using optimizer: {}; with scheduler: {}'.format(self.optimizer, self.scheduler))

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

    def train_step(self, epoch):
        '''
        1 epoch training
        '''
        phase = 'train'
        running_loss = 0.0
        scaler = GradScaler()

        # Set model to training mode
        self.model.train()
        avg_loss = 0.0
        avg_output_std = 0.0
        iters = len(self.dataloaders[phase])
        # Iterate over data.
        for batch_idx, batch in enumerate(self.dataloaders[phase]):
        # for batch, _ in self.dataloaders[phase]:
            # # zero the parameter gradients for every mini batch
            # self.optimizer.zero_grad()
            self.model.zero_grad()
            if self.SSL is None:
                # with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):
                x0, labels = batch
                x0 = x0.to(self.device)
                labels = labels.to(self.device)
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():
                        # make prediction
                        outputs = self.model(x0)
                        # compute loss
                        loss = self.criterion(torch.flatten(outputs), labels.data)

            elif self.SSL == 'SimCLR' or self.SSL == 'VICReg' or self.SSL == 'SimSiam':
                # with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):
                (x0, x1), _, _ = batch
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():
                        # make prediction
                        if self.SSL == 'SimCLR' or self.SSL == 'VICReg':
                            z0 = self.model(x0)
                            z1 = self.model(x1)
                            # compute loss
                            loss = self.criterion(z0, z1)
                        elif self.SSL == 'SimSiam':
                            z0, p0 = self.model(x0)
                            z1, p1 = self.model(x1)
                            # compute loss
                            loss = 0.5 * ((self.criterion(z0, p1) + self.criterion(z1, p0)))
            elif self.SSL == 'VICRegL' or self.SSL == 'VICRegLConvNext':
                views_and_grids = [x.to(self.device) for x in batch[0]]
                # print(self.model)
                views = views_and_grids[:len(views_and_grids)//2]
                grids = views_and_grids[len(views_and_grids) // 2:]

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():
                        # make prediction
                        features = [self.model(view) for view in views]

                        # compute loss
                        global_view_features = features[:2]
                        global_view_grids = grids[:2]
                        local_view_features = features[2:]
                        local_view_grids = grids[2:]

                        loss = self.criterion(
                            global_view_features=global_view_features,
                            global_view_grids=global_view_grids,
                            local_view_features=local_view_features,
                            local_view_grids=local_view_grids,
                        )
            # statistics
            # running_loss += loss.item() * x0.size(0)
            running_loss += loss.item()
            # running_loss += loss.detach()
            # print(running_loss)
            # update gradients
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)  # invokes optimizer.step() if no inf/NaN found
            scaler.update()

            # monitor collapse for SimSiam
            if self.SSL == 'SimSiam' or self.SSL == 'VICReg' or self.SSL == 'SimCLR':# or self.SSL == 'VICRegL':
                # calculate the per-dimension standard deviation of the outputs
                # we can use this later to check whether the embeddings are collapsing
                if self.SSL == 'SimSiam':
                    output = p0.detach()
                # elif self.SSL == 'VICRegL':
                #     output = [feature.detach() for feature in features]
                else:
                    output = z0.detach()
                output = torch.nn.functional.normalize(output, dim=1) # i.e. output has length 1

                output_std = torch.std(output, 0)
                output_std = output_std.mean()
                # use moving averages to track the loss and standard deviation
                w = 0.9
                avg_loss = w * avg_loss + (1 - w) * loss.item()
                avg_output_std = w * avg_output_std + (1 - w) * output_std.item()
                # the level of collapse is large if the standard deviation of the l2
                # normalized output is much smaller than 1 / sqrt(dim)
                # out_dim = 2048 if self.SSL == 'VICReg' else 128
                # out_dim = 2048 if self.SSL == 'VICReg' else 512
                out_dim = 512
                collapse_level = max(0.0, 1 - math.sqrt(out_dim) * avg_output_std)

        # epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
        epoch_loss = running_loss / (len(self.dataloaders[phase].dataset) / self.batch_size)

        # step on lr scheduler
        if self.scheduler is not None:
            self.scheduler.step(epoch)
            # self.scheduler.step(epoch + batch_idx / iters)
            self.lrs.append(self.scheduler.get_last_lr())

        if self.SSL == 'SimSiam' or self.SSL == 'VICReg' or self.SSL == 'SimCLR':# or self.SSL == 'VICRegL' :
            # print intermediate results
            print(
                f"Epoch-Loss = {epoch_loss:.2f} | "
                f"Mov_Avg_Loss = {avg_loss:.2f} | "
                f"Collapse Level: {collapse_level:.2f} / 1.00"
            )
        else:
            print('{} Avg Epoch-Loss: {:.4f}'.format(phase, epoch_loss))
        # return_loss = avg_loss if (self.SSL == 'SimSiam' or self.SSL == 'VICReg') else epoch_loss
        # return return_loss
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
        for batch_idx, batch in enumerate(self.dataloaders[phase]):
            if self.SSL is None:
                x0, labels = batch
                x0 = x0.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    # make prediction
                    outputs = self.model(x0)
                    # compute loss
                    # if self.training_response_standardizer is not None:
                    #     loss = self.criterion((torch.flatten(outputs)*self.training_response_standardizer['std'])+self.training_response_standardizer['mean'], labels.data)
                    # else:
                    #     loss = self.criterion(torch.flatten(outputs), labels.data)
                    loss = self.criterion(torch.flatten(outputs), labels.data)
            elif self.SSL == 'SimCLR' or self.SSL == 'VICReg' or self.SSL == 'SimSiam':
                (x0, x1), _, _ = batch
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)
                with torch.no_grad():
                    # make prediction
                    if self.SSL == 'SimCLR' or self.SSL == 'VICReg':
                        z0 = self.model(x0)
                        z1 = self.model(x1)
                        # compute loss
                        loss = self.criterion(z0, z1)
                    elif self.SSL == 'SimSiam':
                        z0, p0 = self.model(x0)
                        z1, p1 = self.model(x1)
                        # compute loss
                        loss = 0.5 * ((self.criterion(z0, p1) + self.criterion(z1, p0)))
            elif self.SSL == 'VICRegL' or self.SSL == 'VICRegLConvNext':
                views_and_grids = [x.to(self.device) for x in batch[0]]
                views = views_and_grids[:len(views_and_grids)//2]
                grids = views_and_grids[len(views_and_grids) // 2:]
                with torch.no_grad():
                    # make prediction
                    features = [self.model(view) for view in views]
                    # compute loss
                    loss = self.criterion(
                        global_view_features=features[:2],
                        global_view_grids=grids[:2],
                        local_view_features=features[2:],
                        local_view_grids=grids[2:],
                    )
            running_loss += loss.item()

        epoch_loss = running_loss / (len(self.dataloaders[phase].dataset)/self.batch_size)
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        return epoch_loss

    def predict(self, phase:str = 'test', predict_embeddings:bool=False):
        '''
        prediction on dataloader[phase: str] (default: test set)
        '''
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluate mode

        # for each fold store labels and predictions
        local_preds = []
        local_labels = []
        # Iterate over data.
        if not predict_embeddings:
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
        else:
            for inputs, labels in self.dataloaders[phase]:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    # make prediction
                    y_hat = self.model.backbone(inputs).flatten(start_dim=1)
                local_preds.append(y_hat)
                local_labels.extend(labels.detach().cpu().numpy())
            local_preds = torch.cat(local_preds, dim=0)
            local_preds = local_preds.cpu().numpy()

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

        self.lrs = []

        self.model.to(self.device)
        self.model.zero_grad() # zero model gradient across all optimizer
        for epoch in range(num_epochs):
            if patience > 0:
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))
                print('-' * 10)
                for phase in ['train', 'val']:
                    if phase == 'train':
                        epoch_loss = self.train_step(epoch=epoch)
                        train_mse_history.append(epoch_loss)
                    if phase == 'val':
                        epoch_loss = self.test(phase='val')
                        test_mse_history.append(epoch_loss)

                        if epoch > 1:
                            ## check early stopping
                            # if ((best_loss - epoch_loss) / best_loss) <= min_delta:
                            delta = ((best_loss - epoch_loss) / best_loss)
                            # flip sign of delta for SimSiam, bc. of negative cosine similarity
                            if self.SSL == 'SimSiam': delta = -1 * delta
                            if delta <= min_delta:
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
        print('\tTry loading trained model from: {}'.format(checkpoint_path))
        if os.path.exists(checkpoint_path):
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

