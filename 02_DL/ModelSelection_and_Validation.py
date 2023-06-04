# -*- coding: utf-8 -*-
"""
Wrapper class model selection and validation
"""

# Built-in/Generic Imports

# Libs
import time, gc, os

import pandas as pd

import torch
import ray
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

import torch.nn as nn
# Own modules
from TuneYieldRegressor import TuneYieldRegressor
from RGBYieldRegressor_Trainer import RGBYieldRegressor_Trainer

__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'

class ModelSelection_and_Validation:
    def __init__(self,
                 num_folds,
                 pretrained,
                 this_output_dir,
                 datamodule,
                 training_response_normalization,
                 validation_strategy,
                 patch_no,
                 seed,
                 num_epochs,
                 patience,
                 min_delta,
                 min_epochs,
                 momentum,
                 architecture,
                 only_tune_hyperparameters,
                 device,
                 workers,
                 tune_epochs=100,
                 ):

        self.num_folds = num_folds
        self.pretrained = pretrained
        self.this_output_dir = this_output_dir
        self.datamodule = datamodule
        self.training_response_normalization = training_response_normalization
        self.validation_strategy = validation_strategy
        self.patch_no = patch_no
        self.seed = seed
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.momentum = momentum
        self.architecture =architecture
        self.model_wrapper = None
        self.only_tune_hyperparameters = only_tune_hyperparameters
        self.device = device
        self.workers = workers
        self.tune_epochs = tune_epochs

        # init with None
        self.lr = self.wd = self.batch_size = None

    def load_hyperparameters_if_exist(self, k, strategy=''):
        '''
        load tuned hyper params if exist to self.lr/wd/batch_size
        :param k: fold of Cross validation
        :param strategy: specifies training phase for SSL and defaults to '' otherwise
        :return: lr, wd and batch_size if csv exists, None for each variable otherwise
        '''
        # for SSL pretraining, add training phase
        if strategy != '':
            strategy = '_' + strategy

        # load tuned hyper params if exist
        filename = os.path.join(self.this_output_dir, 'hyper_df_' + str(k) + strategy + '.csv')
        if os.path.exists(filename):
            column_names = ['lr', 'wd', 'batch_size']
            df = pd.read_csv(filename, usecols=column_names)
            self.lr = float(df['lr'][0])
            self.wd = float(df['wd'][0])
            self.batch_size = int(df['batch_size'][0])
            return True
        return False

    def save_hyperparameters(self, k, strategy=''):
        # for SSL pretraining, add training phase
        if strategy != '':
            strategy = '_' + strategy
        hyper_df = pd.DataFrame({'lr': [self.lr],
                                 'wd': [self.wd],
                                 'batch_size': [self.batch_size],
                                 })
        hyper_df.to_csv(os.path.join(self.this_output_dir, 'hyper_df_' + str(k) + strategy + '.csv'), encoding='utf-8')

    def get_state_dict_if_exists(self, this_output_dir, k: int, strategy: str = ''):
        '''
        return the state dict of a trained model given path and fold
        :param this_output_dir: path to model
        :param k: number of fold
        :param strategy: pretraining strategy'self-supervised', 'domain-tuning' or '' for none
        :return: state dict of model, or None if it does not exist
        '''
        state_dict = None
        if strategy != '':
            strategy = '_' + strategy

        if os.path.exists(os.path.join(this_output_dir, 'model_f' + str(k) + strategy + '.ckpt')):
            print('initializing model with pretrained weights')
            checkpoint_path = os.path.join(this_output_dir, 'model_f' + str(k) + strategy + '.ckpt')
            if torch.cuda.is_available():
                state_dict = torch.load(checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        return state_dict

    def tune_hyperparameters(self,
                             datamodule=None,
                             tune_fc_only: bool= None,
                             this_output_dir: str = None,
                             patch_no: int = None,
                             seed: int = 42,
                             criterion=None,
                             tune_name: str = None,
                             state_dict=None,
                             ):
        # tune_name[strategy] = 'Tuning_{}_{}_all_bayes_L1_ALB_TL-FC_f{}_{}_{}'.format(self.architecture, validation_strategy, k, patch_no, strategy)

        param_space = {
            "lr": tune.loguniform(1e-6, 1e-1),
            "wd": tune.uniform(0, 5 * 1e-3),
            "batch_size": tune.choice([4, 8, 16, 32, 64, 128, 256, 512]),
        }
        bohb_hyperband = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=100,
            reduction_factor=4,
            stop_last_trials=False,
        )
        # Bayesian Optimization HyperBand -> terminates bad trials + Bayesian Optimization
        bohb_search = TuneBOHB(metric='val_loss',
                               mode='min',
                               seed=seed, )
        bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=4)

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    TuneYieldRegressor,
                    momentum=self.momentum,
                    patch_no=patch_no,
                    architecture=self.architecture,
                    tune_fc_only=tune_fc_only,
                    pretrained=self.pretrained,
                    datamodule=datamodule,
                    criterion=criterion,
                    device=self.device,
                    workers=self.workers,
                    training_response_standardizer=datamodule.training_response_standardizer,
                    state_dict=state_dict,
                ),
                {"gpu": self.workers}),
            param_space=param_space,
            tune_config=tune.TuneConfig(
                metric='val_loss',
                mode='min',
                search_alg=bohb_search,
                scheduler=bohb_hyperband,
                num_samples=20,
            ),
            run_config=ray.air.config.RunConfig(
                # checkpoint_config=ray.air.config.CheckpointConfig(checkpoint_freq=10,),
                failure_config=ray.air.config.FailureConfig(max_failures=5),
                stop={"training_iteration": self.tune_epochs},
                name=tune_name,
                local_dir=this_output_dir, )
        )

        analysis = tuner.fit()
        # torch.save(analysis.get_dataframe(filter_metric="val_loss", filter_mode="min"), os.path.join(this_output_dir, 'analysis_f{}_{}.ray'.format(k, strategy)))
        best_result = analysis.get_best_result(metric="val_loss", mode="min")
        lr = best_result.config['lr']
        wd = best_result.config['wd']
        batch_size = best_result.config['batch_size']
        print('\tbest config: ', analysis.get_best_result(metric="val_loss", mode="min"))

        return lr, wd, batch_size

    def train_and_tune_OneStrategyModel(self, tune_fc_only:bool, criterion, start=0):
        '''
        Tune hyperparamters and rain a models in a 4-fold cross validation approach. Type of cross validation is to be specified in the datamodule.
        :param tune_fc_only: True if only FC layers has gradient computation enabled, Conv layers are frozen
        :param criterion: MSE loss + L2 regularization (nn.MSELoss) or MAE loss + L1 regularization (nn.L1Loss)
        :param start: beginning fold [0..3]
        :return:
        '''
        for k in range(start, self.num_folds):
            print('#' * 60)
            print('Fold: {}'.format(k))
            # reinitialize hyperparams for this fold
            self.lr = self.wd = self.batch_size = None

            # initialize trainer and architecture
            self.model_wrapper = RGBYieldRegressor_Trainer(
                pretrained=self.pretrained,
                tune_fc_only=tune_fc_only,
                architecture=self.architecture,
                criterion=criterion,
                device=self.device,
                workers=self.workers
            )
            # load weights and skip rest of the method if already trained
            if self.model_wrapper.load_model_if_exists(model_dir=self.this_output_dir, strategy=None, k=k):
                continue
            self._train_and_tune_OneFold_OneStrategyModel(tune_fc_only=tune_fc_only,
                                                          criterion=criterion,
                                                          k=k,
                                                          state_dict=None)

    def train_and_tune_SSL(self, start:int=0):
        '''
        Tune hyperparamters and rain a self-supervised models in a 4-fold cross validation approach, i.e. first pre-training is performed on the input data to predict the average mean pixel value
        within a sample.
        Type of cross validation is to be specified in the datamodule.
        :param start: beginning fold [0..3]
        :return:
        '''

        training_strategy = ['self-supervised','domain-tuning']
        training_strategy_params = {
            training_strategy[0]: {
                'tune_fc_only': False,
                'fake_labels': True,
                'augmentation': True,
                'lr': None,
                'wd': None,
                'batch_size': None,
            },
            training_strategy[1]: {
                'tune_fc_only': True,
                'fake_labels': False,
                'augmentation': True,
                'lr': None,
                'wd': None,
                'batch_size': None,
            }
        }
        tune_name = dict()
        tune_name['self-supervised'] = ''
        tune_name['domain-tuning'] = ''

        criterion = {training_strategy[0]: nn.MSELoss(reduction='mean'),
                     training_strategy[1]: nn.MSELoss(reduction='mean'),
                     # training_strategy[1]: nn.L1Loss(reduction='mean'),
                     }

        # save datamodule dict, s.t. we can override self.datamodule with the training strategy respective datamodule
        datamodules = self.datamodule

        for k in range(start, self.num_folds):

            # reinitialize hyperparams for this fold as None
            self.lr = self.wd = self.batch_size = None

            # initialize trainer and architecture
            self.model_wrapper = RGBYieldRegressor_Trainer(
                pretrained=self.pretrained,
                tune_fc_only=training_strategy_params['self-supervised']['tune_fc_only'],
                architecture=self.architecture,
                criterion=criterion['self-supervised'],
                device=self.device,
                workers=self.workers
            )

            for strategy in training_strategy:
                print('#' * 60)
                print('Fold: {}'.format(k))

                # setup datamodules for the respective training strategy
                self.datamodule = datamodules[strategy]

                # load weights and skip rest of the method if already trained
                if self.model_wrapper.load_model_if_exists(model_dir=self.this_output_dir, strategy=strategy, k=k):
                    continue

                # load pretrained model weights for hyper parameter tuning
                if strategy == 'domain-tuning':
                    # get and initialize model with pre-trained, self-supervised weights
                    state_dict = self.get_state_dict_if_exists(this_output_dir=self.this_output_dir, k=k, strategy='self-supervised')
                    self.model_wrapper.model.load_state_dict(state_dict)

                    # freeze conv layers and reinitialize FC layer
                    self.model_wrapper.reinitialize_fc_layers()
                    self.model_wrapper.disable_all_but_fc_grads()
                else:
                    state_dict = None

                self._train_and_tune_OneFold_OneStrategyModel(tune_fc_only=training_strategy_params[strategy]['tune_fc_only'],
                                                              criterion=criterion[strategy],
                                                              k=k,
                                                              state_dict=state_dict,
                                                              strategy=strategy)



    def _train_and_tune_OneFold_OneStrategyModel(self, tune_fc_only, criterion, k, state_dict=None, strategy: str = ''):

        if strategy != '':
            strategy = '_' + strategy

        # sample data into folds according to provided in validation strategy
        self.datamodule.setup_fold(fold=k, training_response_standardization=self.training_response_normalization)

        # tune hyper parameters #######################
        tune_name = 'Tuning_{}_{}_all_bayes_L1_ALB_f{}{}_{}'.format(self.architecture, self.validation_strategy, k, strategy, self.patch_no)

        # load hyper params if already tuned
        if not self.load_hyperparameters_if_exist(k=k, strategy=strategy):
            self.lr, self.wd, self.batch_size = self.tune_hyperparameters(datamodule=self.datamodule,
                                                                    tune_fc_only=tune_fc_only,
                                                                    this_output_dir=self.this_output_dir,
                                                                    patch_no=self.patch_no,
                                                                    seed=self.seed,
                                                                    criterion=criterion,
                                                                    tune_name=tune_name,
                                                                    state_dict=state_dict,
                                                                    )
            self.save_hyperparameters(k=k, strategy=strategy)

        if not self.only_tune_hyperparameters:
            ###################### Training #########################
            # set hyper parameters for model wrapper and data module
            self.datamodule.set_batch_size(batch_size=self.batch_size)
            self.model_wrapper.set_hyper_parameters(lr=self.lr, wd=self.wd, batch_size=self.batch_size)

            # build and set data loader dict
            dataloaders_dict = {'train': self.datamodule.train_dataloader(), 'val': self.datamodule.val_dataloader(),
                                # 'test': datamodule.test_dataloader(),
                                }
            self.model_wrapper.set_dataloaders(dataloaders=dataloaders_dict)

            # update optimizer to new hyper parameter set
            self.model_wrapper.set_optimizer()

            # data parallelize and send model to device:: only if not already done -> if strategy '' or 'self-supervised'
            if strategy == '' or strategy == 'self-supervised':
                self.model_wrapper.parallize_and_to_device()

            # measure training time
            self.model_wrapper.start_timer()

            # Train and evaluate
            print('training for {} epochs'.format(self.num_epochs))
            self.model_wrapper.train(patience=self.patience,
                                min_delta=self.min_delta,
                                num_epochs=self.num_epochs,
                                min_epochs=self.min_epochs,
                                )


            # save best model
            torch.save(self.model_wrapper.model.state_dict(), os.path.join(self.this_output_dir, 'model_f' + str(k) + strategy + '.ckpt'))
            run_time = self.model_wrapper.end_timer_and_get_time('\nEnd training for fold {}'.format(k))

            # save training statistics
            df = pd.DataFrame({'train_loss': self.model_wrapper.train_mse_history,
                               'val_loss': self.model_wrapper.test_mse_history,
                               'best_epoch': self.model_wrapper.best_epoch,
                               'time': run_time,
                               })
            df.to_csv(os.path.join(self.this_output_dir, 'training_statistics_f' + str(k) + strategy + '.csv'), encoding='utf-8')


