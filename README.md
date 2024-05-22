# MOMBO
Official implementation of Deterministic Uncertainty Propagation for Improved Model-Based Offline Reinforcement Learning

## Requirements

To install all the required dependencies:

1. Install MuJoCo engine, which can be downloaded from [here](https://mujoco.org/download).
2. Install Python packages listed in `requirements.txt` using `pip install -r requirements.txt`. You should specify the version of `mujoco-py` in `requirements.txt` depending on the version of MuJoCo engine you have installed.
3. Manually download and install `d4rl` package from [here](https://github.com/rail-berkeley/d4rl).

## Usage

Just run `train.py` with specifying the task name. Other hyperparameters are automatically loaded from `config`. 

```bash
python train.py --task [TASKNAME]
```
