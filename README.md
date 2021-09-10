# DeepVecFont

This is the official Tensorflow implementation of the paper:

Yizhi Wang and Zhouhui Lian. DeepVecFont: Synthesizing High-quality Vector Fonts via Dual-modality Learning. SIGGRAPH 2021 Asia. 2021c.

## Installation

### Requirement

- **python 3.9**
- **Pytorch 1.9** (it may work on some lower versions, but not tested)

Please use [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) to build the environment:

### Install diffvg

## Training and Testing


to train our main model, run
```
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --experiment_name dvf --model_name main_model
```

to train the neural rasterizer
CUDA_VISIBLE_DEVICES=0 python train_nr.py --mode train --experiment_name dvf --model_name neural_raster

```
CUDA_VISIBLE_DEVICES=0 python train_sr.py --mode train --experiment_name image_ss --model_name image_sr
```

# to test our main model, run

CUDA_VISIBLE_DEVICES=0 python test_sf.py --mode test --experiment_name argmax_v1.0 --model_name main_model --test_epoch 625 --batch_size 1

