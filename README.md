# SNU_robotvision_2024_fall
Seoul National University's robot vision class 2024 fall team project

# How to use
## Setup
conda env 가 없는 경우
`conda create -n "환경 이름" python=3.8`
conda env 가 있는 경우 (python 3.8 아니면 재고 필요)
`conda env export > environment.yaml`

## To train
Put your network name on it.
`python train_[network]`

You have to download the GTA5 dataset first. - Call Hyunho.
You have to change the path to train dataset.
There is no additional validation set on source model training.
