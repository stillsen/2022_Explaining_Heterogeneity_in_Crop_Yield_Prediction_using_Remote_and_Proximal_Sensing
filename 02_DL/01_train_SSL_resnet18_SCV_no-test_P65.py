# -*- coding: utf-8 -*-
"""
script to train model
"""

# Built-in/Generic Imports
import os, random, time, gc


# Libs
from ctypes import c_int

import numpy as np
import pandas as pd

import time
from copy import deepcopy

import ray
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB


import torch
from torch import nn

import warnings
# Own modules
from PatchCROPDataModule import PatchCROPDataModule
from RGBYieldRegressor import RGBYieldRegressor
from TuneYieldRegressor import TuneYieldRegressor

from directory_listing import output_dirs, data_dirs, input_files_rgb


def start_timer(device=None):
    global start_time
    gc.collect()
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated(device=device)
        torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_get_time(local_msg):
    if device == torch.device("cuda"):
        torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {} sec".format(end_time - start_time))
    print('Max memory used by tensors = {} bytes'.format(torch.cuda.max_memory_allocated()))
    return end_time - start_time


__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'


if __name__ == "__main__":
    # ray.init(local_mode=True)
    seed = 42
    # seed_everything(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    ## HYPERPARAMETERS
    num_epochs = 1000
    # lr_1 = None
    # lr_2 = None
    lr_finetuning = None
    # lr_finetuning = 0.0005
    momentum = 0.9 # (Krizhevsky et al.2012)
    # wd = 0.0005 # (Krizhevsky et al.2012)
    # wd = 0.003593991109916679
    # wd_1 = None
    # wd_2 = None
    classes = 1
    # batch_size_1 = None
    # batch_size_2 = None
    # batch_size = 256 # tuning 1
    num_folds = 4#9 # ranom-CV -> 1
    min_delta = 0.01 # aka 1%
    patience = 10
    min_epochs = 1000
    # repeat_trainset_ntimes_1 = 1
    # repeat_trainset_ntimes_2 = 10

    # patch_no = 73
    patch_no = 65
    stride = 30 # 20 is too small
    # architecture = 'baselinemodel'
    # architecture = 'densenet'
    architecture = 'resnet18'
    # augmentation_1 = False
    # augmentation_2 = True
    # tune_fc_only_1 = False
    # tune_fc_only_2 = True
    pretrained = False
    features = 'RGB'
    # features = 'RGB+'
    num_samples_per_fold = None # subsamples? -> None means do not subsample but take whole fold
    validation_strategy = 'SCV_no_test' # SHOV => Spatial Hold Out Validation; SCV => Spatial Cross Validation; SCV_no_test; RCV => Random Cross Validation
    # scv = False
    # fake_labels_1 = True
    # fake_labels_2 = False
    # training_response_normalization = True
    training_response_normalization = False


    this_output_dir = output_dirs[patch_no]+'_'+architecture+'_'+validation_strategy+'_SSL_ALB'+'_E'+str(num_epochs)+'_resetW'

    # check if exists, -> error,
    # else create
    if not os.path.exists(this_output_dir):
        print('creating: \t {}'.format(this_output_dir))
        os.mkdir(this_output_dir)
    else:
        warnings.warn("{} directory exists. WILL OVERRIDE.".format(this_output_dir))

    print('working on patch {}'.format(patch_no))
    # loop over folds, last fold is for testing only

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('working on device %s' % device)
    if device == 'cpu':
        workers = os.cpu_count()
    else:
        workers = 1#torch.cuda.device_count()
        print('\twith {} workers'.format(workers))

    print('Setting up data in {}'.format(data_dirs[patch_no]))

    # dictionary for training strategy::
    # 1) self-supervised pretraining
    # 2) domain-tuning
    training_strategy = ['self-supervised','domain-tuning']
    training_strategy_params = {
        training_strategy[0]: {
                'tune_fc_only': False,
                'fake_labels': True,
                'augmentation' : True,
                'lr' : None,
                'wd' : None,
                'batch_size' : None,
                'repeat_trainset_ntimes' : 1,
        },
        training_strategy[1]: {
            'tune_fc_only': True,
            'fake_labels': False,
            'augmentation' : True,
            'lr' : None,
            'wd' : None,
            'batch_size' : None,
            'repeat_trainset_ntimes' : 1,
        }
    }
    tune_name = dict()
    tune_name['self-supervised'] = ''
    tune_name['domain-tuning'] = ''

    criterion = {training_strategy[0]: nn.MSELoss(reduction='mean'),
                 training_strategy[1] : nn.L1Loss(reduction='mean')}

    datamodule = dict()
    datamodule['self-supervised'] = PatchCROPDataModule(input_files=input_files_rgb[patch_no],
                                                        patch_id=patch_no,
                                                        data_dir=data_dirs[patch_no],
                                                        stride=stride,
                                                        workers=workers,
                                                        augmented=training_strategy_params['self-supervised']['augmentation'],
                                                        input_features=features,
                                                        batch_size=training_strategy_params['self-supervised']['batch_size'],
                                                        validation_strategy=validation_strategy,
                                                        fake_labels=training_strategy_params['self-supervised']['fake_labels'],
                                                        )
    datamodule['self-supervised'].prepare_data(num_samples=num_samples_per_fold)
    datamodule['domain-tuning'] = PatchCROPDataModule(input_files=input_files_rgb[patch_no],
                                                      patch_id=patch_no,
                                                      data_dir=data_dirs[patch_no],
                                                      stride=stride,
                                                      workers=workers,
                                                      augmented=training_strategy_params['domain-tuning']['augmentation'],
                                                      input_features=features,
                                                      batch_size=training_strategy_params['domain-tuning']['batch_size'],
                                                      validation_strategy=validation_strategy,
                                                      fake_labels=training_strategy_params['domain-tuning']['fake_labels'],
                                                      )
    datamodule['domain-tuning'].prepare_data(num_samples=num_samples_per_fold)


    for k in range(num_folds):
        print('#'*60)
        run_time=0
        # setup data according to folds
        # quadrants:
        # 0 1
        # 2 3
        # test val train   | fold
        #  0    1   {2,3}  |  0
        #  1    3   {0,2}  |  1
        #  3    2   {0,1}  |  2
        #  2    0   {1,3}  |  3        # save training statistics
        train_loss = dict()
        val_loss = dict()
        best_epoch = dict()
        # 1) self-supervised pretraining, 2) domain-tuning
        for strategy in training_strategy:
            # sample data in folds & augment and standardize
            # datamodule[strategy].setup_fold(training_response_standardization=training_response_normalization)
            # dataloaders_dict = {'train': datamodule[strategy].train_dataloader(), 'val': datamodule[strategy].val_dataloader()}
            datamodule[strategy].setup_fold(fold=k, training_response_standardization=training_response_normalization, duplicate_trainset_ntimes=training_strategy_params[strategy]['repeat_trainset_ntimes'])
            # datamodule.create_debug_samples(n=20)
            print('Fold: {} - {}'.format(k,strategy))

            # is this model already trained and saved? -> load and skip rest
            if os.path.exists(os.path.join(this_output_dir, 'model_f' + str(k) +'_'+ strategy+'.ckpt')):
                print('\tLoading trained model: {}'.format(os.path.join(this_output_dir, 'model_f' + str(k) +'_'+ strategy+'.ckpt')))
                checkpoint_path = os.path.join(this_output_dir, 'model_f' + str(k) +'_'+ strategy+'.ckpt')
                # load trained model weights with respect to possible parallelization and device
                if torch.cuda.is_available():
                    state_dict = torch.load(checkpoint_path)
                else:
                    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                if workers > 1:
                    model_wrapper.model = nn.DataParallel(model_wrapper.model)

                model_wrapper.model.load_state_dict(state_dict)
                continue

            ## SELF-SUPERVISED PRE-TRAINING

            ###### tune hyper parameters #######################
            ###### FROZEN CONV ######### tune hyper parameters #######################
            tune_name[strategy] = 'Tuning_{}_{}_all_bayes_L1_ALB_TL-FC_f{}_{}_TR{}_{}'.format(architecture, validation_strategy, k, patch_no, training_strategy_params[strategy]['repeat_trainset_ntimes'], strategy)
            print('\t{}-tuning'.format(strategy))

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

            # if domain tuning, load and pass the state dict of the pretrained model for hyper param tuning
            state_dict=None
            if os.path.exists(os.path.join(this_output_dir, 'model_f' + str(k) +'_'+ strategy+'.ckpt')):
                checkpoint_path = os.path.join(this_output_dir, 'model_f' + str(k) +'_'+ strategy+'.ckpt')
                if torch.cuda.is_available():
                    state_dict = torch.load(checkpoint_path)
                else:
                    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(
                        TuneYieldRegressor,
                        momentum=momentum,
                        patch_no=patch_no,
                        architecture=architecture,
                        tune_fc_only=training_strategy_params[strategy]['tune_fc_only'],
                        pretrained=pretrained,
                        datamodule=datamodule[strategy],
                        criterion=criterion[strategy],
                        device=device,
                        workers=workers,
                        training_response_standardizer=datamodule[strategy].training_response_standardizer,
                        state_dict=state_dict,
                    ),
                    {"gpu": workers}),
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
                    stop={"training_iteration": 100},
                    name=tune_name[strategy],
                    local_dir=this_output_dir, )
            )

            # # does hyper-parameter tuning for this configuration already exist?
            # if os.path.exists(os.path.join(this_output_dir, 'analysis_f{}_{}.ray'.format(k,strategy))):
            #     print('\tLoading hyper-parameter tuning : {}'.format(os.path.join(this_output_dir, 'analysis_f{}_{}.ray'.format(k,strategy))))
            #     # analysis = torch.load(os.path.join(this_output_dir, 'analysis_f{}_{}.ray'.format(k,strategy)))
            #     tuner = tune.Tuner.restore(os.path.join(this_output_dir, tune_name[strategy]))
            #     analysis = tuner.get_results()
            # else: # no? -> tune

            # if k ==1:
            #     ray.init(_temp_dir='/beegfs/stiller/PatchCROP_all/tmp/ray')
            #     tuner.restore(path=os.path.join(this_output_dir,tune_name))
            analysis = tuner.fit()
            torch.save(analysis.get_dataframe(filter_metric="val_loss", filter_mode="min"), os.path.join(this_output_dir, 'analysis_f{}_{}.ray'.format(k,strategy)))


            print('\tbest config: ', analysis.get_best_result(metric="val_loss", mode="min"))
            best_result = analysis.get_best_result(metric="val_loss", mode="min")
            training_strategy_params[strategy]['lr'] = best_result.config['lr']
            training_strategy_params[strategy]['wd'] = best_result.config['wd']
            training_strategy_params[strategy]['batch_size'] = best_result.config['batch_size']

            ######  train model #######################
            datamodule[strategy].set_batch_size(batch_size=training_strategy_params[strategy]['batch_size'])
            dataloaders_dict = {'train': datamodule[strategy].train_dataloader(), 'val': datamodule[strategy].val_dataloader(),
                                # 'test': datamodule[strategy].test_dataloader(),
                                }
            start_timer()

            if strategy == 'self-supervised':
                model_wrapper = RGBYieldRegressor(dataloaders=dataloaders_dict,
                                                  device=device,
                                                  lr=training_strategy_params[strategy]['lr'],
                                                  momentum=momentum,
                                                  wd=training_strategy_params[strategy]['wd'],
                                                  # k=num_folds,
                                                  pretrained=pretrained,
                                                  tune_fc_only=training_strategy_params[strategy]['tune_fc_only'],
                                                  model=architecture,
                                                  training_response_standardizer=datamodule[strategy].training_response_standardizer,
                                                  criterion=criterion[strategy],
                                                  )
                if workers > 1:
                    model_wrapper.model = nn.DataParallel(model_wrapper.model)
                model_wrapper.model.to(device)

            else:
                # reintialize fc layer's weights
                model_wrapper.reset_SSL_fc_weights()
                # update dataloaders
                model_wrapper.set_dataloaders(dataloaders=dataloaders_dict)
                # update optimizer
                model_wrapper.set_optimizer(lr=training_strategy_params[strategy]['lr'],
                                            momentum=momentum,
                                            wd=training_strategy_params[strategy]['wd'],
                                            )
                # update criterion
                model_wrapper.set_criterion(criterion=criterion[strategy])
                # disable gradient computation for the first  layers
                model_wrapper.disable_all_but_fc_grads()
            # model_wrapper.print_children_require_grads()
            # Send the model to GPU

            # load weights, if model already exists, else train
            if os.path.exists(os.path.join(this_output_dir, 'model_f' + str(k) +'_'+ strategy+'.ckpt')):
                print('\tLoading trained model: {}'.format(os.path.join(this_output_dir, 'model_f' + str(k) +'_'+ strategy+'.ckpt')))
                checkpoint_path = os.path.join(this_output_dir, 'model_f' + str(k) +'_'+ strategy+'.ckpt')
                # load trained model weights with respect to possible parallelization and device
                if torch.cuda.is_available():
                    state_dict = torch.load(checkpoint_path)
                else:
                    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                if workers > 1:
                    model_wrapper.model = nn.DataParallel(model_wrapper.model)

                model_wrapper.model.load_state_dict(state_dict)
            else:# Train and evaluate
                print('training for {} epochs'.format(num_epochs))
                model_wrapper.train_model(patience=patience,
                                          min_delta=min_delta,
                                          num_epochs=num_epochs,
                                          min_epochs=min_epochs,
                                          )
                run_time = end_timer_and_get_time('\nEnd training for fold {}'.format(k))
                # save best model
                torch.save(model_wrapper.model.state_dict(), os.path.join(this_output_dir, 'model_f' + str(k) +'_'+ strategy+'.ckpt'))

            if not os.path.exists(os.path.join(this_output_dir, 'training_statistics_f' + str(k) +'_'+ strategy + '.csv')):
                print('\tWriting training meta data: {}'.format(os.path.join(this_output_dir, 'training_statistics_f' + str(k) +'_'+ strategy + '.csv')))
                # save training statistics
                df = pd.DataFrame({'train_loss': model_wrapper.train_mse_history,
                                   'val_loss': model_wrapper.test_mse_history,
                                   'best_epoch': model_wrapper.best_epoch,
                                   'time': run_time,
                                   # 'ft_val_loss':ft_val_losses,
                                   # 'ft_train_loss':ft_train_losses,
                                   # 'ft_best_epoch':ft_best_epoch,
                                   })
                df.to_csv(os.path.join(this_output_dir, 'training_statistics_f' + str(k) +'_'+ strategy + '.csv'), encoding='utf-8')

