# SNU_robotvision_2024_fall
Seoul National University's robot vision class 2024 fall team project

# Git
## 등록 및 레포지토리 클론
0. SSH key 등록이 안됐으면 이 [링크] 설명대로 진행
1. git clone git@github.com:HYUNHONOH98/SNU_robotvision_2024_fall.git
2. git config --global user.email "[github 이메일]"
3. git config --global user.name "[githun 이름]"
4. git pull origin main
## 브랜치에서 작업하고 머지하기 (main 에서 절대 작업 금지)
1. git checkout -b [작업할 브랜치 이름.] (ex. UDA_implement)

.. 브랜치에서 작업 .. 작업이 끝나면

2. git stash (지금까지 작업한거 잠깐 임시 저장)
3. git pull origin main
4. git stash pop (임시 저장한 변경사항들 다시 불러오기)
5. git add .
6. git commit -m "[실험한 내용 적기]"
7. git push

여기까지 끝나면 팀원들한테 얘기하기.

# How to use
## Setup
conda env 가 없는 경우

`conda create -n "환경 이름" python=3.8`

conda env 가 있는 경우 (python 3.8 아니면 재고 필요)

`conda env export > environment.yaml`

## To train
Put your network name on it.

`python train_[network]`

You have to download the GTA5 dataset first. - 현호를 불러주세요 보내드림

You have to change the path to train dataset.

There is no additional validation set on source model training.

[링크]: https://docs.github.com/ko/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/