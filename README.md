# SNU_robotvision_2024_fall
Seoul National University's robot vision class 2024 fall team project

# 0. Git
## 0.1. 등록 및 레포지토리 클론
0. SSH key 등록이 안됐으면 이 [링크] 설명대로 진행
1. git clone git@github.com:HYUNHONOH98/SNU_robotvision_2024_fall.git
2. git config --global user.email "[github 이메일]"
3. git config --global user.name "[githun 이름]"
4. git pull origin main
## 0.2. 브랜치에서 작업하고 머지하기 (main 에서 절대 작업 금지)
1. git checkout -b [작업할 브랜치 이름.] (ex. UDA_implement)

.. 브랜치에서 작업 .. 작업이 끝나면

2. git stash (지금까지 작업한거 잠깐 임시 저장)
3. git pull origin main
4. git stash pop (임시 저장한 변경사항들 다시 불러오기)
    > 왜 stash 를 하냐면, main 에 변경 사항이 있는데 그냥 add commit push 하면 충돌이 무조건 남. push 하기 전에는 반드시 main 과 내가 작업하고 있는 브랜치를 항상 최신으로 맞춰주기
5. git add .
6. git commit -m "[실험한 내용 적기]"
7. git push

여기까지 끝나면 팀원들한테 얘기하기.

# 1. How to use
## 1.1. Setup
- conda 설치가 안돼있는 경우 : 자기 상황에 맞게 [해당 문서] 참고해서 miniconda 다운로드

- conda env 가 없는 경우

`conda create -n "환경 이름" python=3.8`

- conda env 가 있는 경우 (python 3.8 아니면 재고 필요)

`conda env export > environment.yaml`

- data/ 폴더를 만들어주기. (데이터셋 다운받을 때 저장할 폴더)

## 1.2. Download the dataset
- **cityscapes**
  1. [cityscapes 링크] 들어가서 회원가입하고, leftImg8bit_trainvaltest.zip (11GB : 사진), gtFine_trainvaltest.zip (241MB : annotation) 다운로드
  2. unzip gtFine_trainvaltest.zip <- OS (윈도우, 우분투 등)에 따라 다를 수 있는데 해보기
  3. data/ 에 옮기기
  4. `python utils/dataset/cityscapes_dataset_converter.py [cityscapes_dataset 경로] --nproc 8` 실행
- **GTA5**
  1. [GTA5 링크] 들어가서 저기 써있는대로 다운받으면 됨. or [GTA5 링크2] 들어가서 직접 눌러서 다운받기
  2. unzip
  3. data/ 에 옮기기
- 확인
  - _examples/_example_dataset.ipynb 들어가서 셀들 실행해보면 잘 다운됐는지 확인 가능.

## 1.3. To train
Put your network name on it.

`python train_[network]`

You have to [download the GTA5, Cityscapes dataset](#download-the-dataset) first.

You have to change the **path to train dataset.**

There is no additional validation set on source model training.

[링크]: https://docs.github.com/ko/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/

[cityscapes 링크]: https://www.cityscapes-dataset.com/downloads/

[GTA5 링크]: https://github.com/sarrrrry/PyTorchDL_GTA5/

[GTA5 링크2]: https://download.visinf.tu-darmstadt.de/data/from_games/

[해당 문서]: https://docs.anaconda.com/miniconda/