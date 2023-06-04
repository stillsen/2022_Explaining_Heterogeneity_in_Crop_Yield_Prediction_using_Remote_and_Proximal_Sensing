# -*- coding: utf-8 -*-
"""
script to train the baseline model following (Nevavuori et al.2019) and (Krizhevsky et al.2012)
"""

# Built-in/Generic Imports
import os, random, time, gc

# Libs
import numpy as np
import pandas as pd

import time
from copy import deepcopy

import ray
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.air import session
from ray.air.checkpoint import Checkpoint


import torch
from torch import nn

import warnings
# Own modules
from PatchCROPDataModule import PatchCROPDataModule
from RGBYieldRegressor_Trainer import RGBYieldRegressor_Trainer
from TuneYieldRegressor import TuneYieldRegressor

from directory_listing import output_dirs, data_dirs, input_files_rgb


__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'

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

if __name__ == "__main__":
    # ray.init(local_mode=True)
    seed = 42
    # seed_everything(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    ## HYPERPARAMETERS
    num_epochs = 2000
    # num_epochs_finetuning = 10
    # lr = 0.001 # (Krizhevsky et al.2012)
    # lr = 0.012234672196538655
    lr = None
    # lr_finetuning = 0.0005
    momentum = 0.9 # (Krizhevsky et al.2012)
    # wd = 0.0005 # (Krizhevsky et al.2012)
    # wd = 0.003593991109916679
    wd = None
    classes = 1
    # batch_size = 16
    # batch_size = None
    batch_size = 128 # tuning 1
    num_folds = 4#9 # ranom-CV -> 1
    min_delta = 0.01 # aka 1%
    patience = 10
    min_epochs = 2000
    duplicate_trainset_ntimes = 1

    # patch_no = 73
    patch_no = 19
    stride = 30 # 20 is too small
    # architecture = 'baselinemodel'
    # architecture = 'densenet'
    architecture = 'resnet18'
    augmentation = True
    tune_fc_only = False
    pretrained = False
    features = 'RGB'
    # features = 'RGB+'
    num_samples_per_fold = None # subsamples? -> None means do not subsample but take whole fold
    validation_strategy = 'SCV_no_test' # SHOV => Spatial Hold Out Validation; SCV => Spatial Cross Validation; SCV_no_test; RCV => Random Cross Validation
    # scv = False
    fake_labels = False
    # training_response_normalization = True
    training_response_normalization = False
    criterion = nn.L1Loss(reduction='mean')

    this_output_dir = output_dirs[patch_no]+'_'+architecture+'_'+validation_strategy+'_L1_ALB_TR'+str(duplicate_trainset_ntimes)+'_E'+str(num_epochs)
    # this_output_dir = output_dirs[patch_no] + '_' + 'SSL' + '_' + validation_strategy + '_grn'

    # check if exists, -> error,
    # else create
    if not os.path.exists(this_output_dir):
        print('creating: \t {}'.format(this_output_dir))
        os.mkdir(this_output_dir)
    else:
        # raise FileExistsError("{} is a directory, cannot create new one".format(this_output_dir))
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
    datamodule = PatchCROPDataModule(input_files=input_files_rgb[patch_no],
                                     patch_id=patch_no,
                                     data_dir=data_dirs[patch_no],
                                     stride=stride,
                                     workers=workers,
                                     augmented=augmentation,
                                     input_features=features,
                                     batch_size=batch_size,
                                     validation_strategy=validation_strategy,
                                     fake_labels=fake_labels,
                                     )
    datamodule.prepare_data(num_samples=num_samples_per_fold)

    # for k in range(1):
    for k in range(num_folds):
        print('#'*60)
        print('Fold: {}'.format(k))

        # setup data according to folds
        # quadrants:
        # 0 1
        # 2 3
        # test val train   | fold
        #  0    1   {2,3}  |  0
        #  1    3   {0,2}  |  1
        #  3    2   {0,1}  |  2
        #  2    0   {1,3}  |  3
        datamodule.setup_fold(fold=k, training_response_standardization=training_response_normalization, duplicate_trainset_ntimes=duplicate_trainset_ntimes)
        # datamodule.create_debug_samples(n=20)
        # datamodule.set_batch_size(batch_size=batch_size)
        # dataloaders_dict = {'train': datamodule.train_dataloader(), 'val': datamodule.val_dataloader(),
        #                     # 'test': datamodule.test_dataloader(),
        #                     }
        ####################### tune hyper parameters #######################3
        if k >= 0:
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
                seed=seed,)
            bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=4)

            tune_name = 'Tuning_{}_{}_all_bayes_L1_ALB_f{}_{}_TR{}'.format(architecture, validation_strategy, k, patch_no, duplicate_trainset_ntimes)
            # analysis = tune.run(tune.with_parameters(
            #                         TuneYieldRegressor,
            #                         momentum=momentum,
            #                         patch_no=patch_no,
            #                         architecture=architecture,
            #                         tune_fc_only=tune_fc_only,
            #                         pretrained=pretrained,
            #                         datamodule=datamodule,
            #                         criterion=criterion,
            #                         device=device,
            #                         ),
            #                     checkpoint_freq=10,
            #                     max_failures=5,
            #                     # stop={"training_iteration" : 20},
            #                     config=param_space,
            #                     resources_per_trial={"gpu": 2},
            #                     metric='val_loss',
            #                     mode='min',
            #                     resume="AUTO",
            #                     search_alg=algo,
            #                     scheduler=bohb_hyperband,
            #                     num_samples=20,
            #                     stop={"training_iteration": 100},
            #                     name=tune_name,
            #                     local_dir=this_output_dir,
            #                     )
            tuner = tune.Tuner(
                                tune.with_resources(
                                    tune.with_parameters(
                                        TuneYieldRegressor,
                                        momentum=momentum,
                                        patch_no=patch_no,
                                        architecture=architecture,
                                        tune_fc_only=tune_fc_only,
                                        pretrained=pretrained,
                                        datamodule=datamodule,
                                        # datamodule=dataloaders_dict,
                                        criterion=criterion,
                                        device=device,
                                        workers=workers,
                                        training_response_standardizer=datamodule.training_response_standardizer
                                        ),
                                    {"gpu": 1}),
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
                                name=tune_name,
                                local_dir=this_output_dir,)
                                )

            # if k ==1:
            #     ray.init(_temp_dir='/beegfs/stiller/PatchCROP_all/tmp/ray')
            #     tuner.restore(path=os.path.join(this_output_dir,tune_name))
            # ray.init(_temp_dir='/beegfs/stiller/PatchCROP_all/tmp/ray')
            analysis = tuner.fit()
            torch.save(analysis, os.path.join(this_output_dir, 'analysis_f{}.ray'.format(k)))
        else:
            analysis = torch.load(os.path.join(this_output_dir, 'analysis_f{}.ray'.format(k)))
        print('\tbest config: ', analysis.get_best_result(metric="val_loss", mode="min"))

        best_result = analysis.get_best_result(metric="val_loss", mode="min")
        lr = best_result.config['lr']
        wd = best_result.config['wd']
        batch_size = best_result.config['batch_size']
        # lr = 0.001
        # wd = 0.0005
        # batch_size = 64
        ###################### Training #########################
        datamodule.set_batch_size(batch_size=batch_size)
        dataloaders_dict = {'train': datamodule.train_dataloader(), 'val': datamodule.val_dataloader(),
                            # 'test': datamodule.test_dataloader(),
                            }
        start_timer()
        model_wrapper = RGBYieldRegressor_Trainer(dataloaders=dataloaders_dict,
                                                  device=device,
                                                  lr=lr,
                                                  momentum=momentum,
                                                  wd=wd,
                                                  # k=num_folds,
                                                  pretrained=pretrained,
                                                  tune_fc_only=tune_fc_only,
                                                  architecture=architecture,
                                                  training_response_standardizer=datamodule.training_response_standardizer,
                                                  criterion=criterion,
                                                  )

        # # Send the model to GPU
        # if torch.cuda.device_count() > 1:
        #     model_wrapper.model = nn.DataParallel(model_wrapper.model)
        model_wrapper.model.to(device)

        # Train and evaluate
        print('training for {} epochs'.format(num_epochs))
        model_wrapper.train(patience=patience,
                            min_delta=min_delta,
                            num_epochs=num_epochs,
                            min_epochs=min_epochs,
                            )
        run_time = end_timer_and_get_time('\nEnd training for fold {}'.format(k))
        # save best model
        torch.save(model_wrapper.model.state_dict(), os.path.join(this_output_dir, 'model_f' + str(k) + '.ckpt'))
        # save training statistics
        df = pd.DataFrame({'train_loss': model_wrapper.train_mse_history,
                           'val_loss': model_wrapper.test_mse_history,
                           'best_epoch': model_wrapper.best_epoch,
                           'time': run_time,
                           # 'ft_val_loss':ft_val_losses,
                           # 'ft_train_loss':ft_train_losses,
                           # 'ft_best_epoch':ft_best_epoch,
                           })
        df.to_csv(os.path.join(this_output_dir, 'training_statistics_f' + str(k) + '.csv'), encoding='utf-8')
