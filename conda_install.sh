#!/bin/bash
conda create --name PL16 python=3.8
conda activate PL16

# @HPC
conda install cudatoolkit=10.0 -c pytorch
conda install pytorch-gpu -c conda-forge
#./Programme/Anaconda3/envs/PL38/bin/pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip
~/anaconda3/envs/PL38/bin/pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip
# end @HPC

conda install -c conda-forge 'arosics>=1.3.0'
#conda install -c conda-forge pytorch-lightning
# conda forge important or conflict
conda install -c conda-forge torchvision
conda install -c conda-forge seaborn
#conda install -c conda-forge ray-tune

# conda install pytorch-ignite -c conda-forge
#conda install pip

#~/Programme/Anaconda3/envs/PL16/bin/pip install ray
#~/Programme/Anaconda3/envs/PL16/bin/pip install hpbandster ConfigSpace
~/Programs/Anaconda3/envs/PL16/bin/pip install ray==1.13.0
~/Programs/Anaconda3/envs/PL16/bin/pip install hpbandster ConfigSpace bayesian-optimization
#./Programme/Anaconda3/envs/PL38/bin/pip install spacv
#./Programme/Anaconda3/envs/PL38/bin/pip install fastai

#--------------------------------------------------------------
conda create -n CUDA python=3.8
conda activate CUDA
conda install -c conda-forge 'arosics>=1.3.0' # needs to be at position 1, otherwise gdal throws errors <3

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

#~/anaconda3/envs/CUDA/bin/pip install ray==1.13.0
~/anaconda3/envs/CUDA/bin/pip install ray
#~/Programs/Anaconda3/envs/CUDA/bin/pip install ray==1.13.0
~/anaconda3/envs/CUDA/bin/pip install hpbandster ConfigSpace bayesian-optimization ray[rllib]
#~/Programs/Anaconda3/envs/PL16/bin/pip install hpbandster ConfigSpace bayesian-optimization ray[rllib]

~/anaconda3/envs/CUDA/bin/pip install -U albumentations
#~/Programs/Anaconda3/envs/PL16/bin/pip install -U albumentations

#-------------------------------------------------------------------
conda create -n PTC118 python=3.9 ipython
conda install pytorch torchvision torchaudio pytorch-cuda=11.8  -c pytorch -c nvidia

~/anaconda3/envs/PTC118/bin/pip install ray==2.2.0
~/anaconda3/envs/PTC118/bin/pip install hpbandster ConfigSpace bayesian-optimization ray[rllib]
~/anaconda3/envs/PTC118/bin/pip install -U albumentations
~/anaconda3/envs/PTC118/bin/pip install lightly

#-------------------------------------------------------------------
conda create -n PTC118P38 python=3.8 ipython
conda activate PTC118P38
conda install -c conda-forge 'arosics>=1.3.0' # needs to be at position 1, otherwise gdal throws errors <3

conda install pytorch torchvision torchaudio pytorch-cuda=11.8  -c pytorch -c nvidia

~/anaconda3/envs/PTC118P38/bin/pip install ray==2.2.0
~/anaconda3/envs/PTC118P38/bin/pip install hpbandster ConfigSpace bayesian-optimization ray[rllib]
~/anaconda3/envs/PTC118P38/bin/pip install -U albumentations
~/anaconda3/envs/PTC118P38/bin/pip install lightly
conda install -c conda-forge umap-learn
conda install -c plotly plotly=5.18.0
