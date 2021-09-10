# DeepVecFont

This is the official Tensorflow implementation of the paper:

Yizhi Wang and Zhouhui Lian. DeepVecFont: Synthesizing High-quality Vector Fonts via Dual-modality Learning. SIGGRAPH 2021 Asia. 2021.

<div align=center>
	<img src="imgs/teaser.svg" width="500"> 
</div>

## Installation

### Requirement

- **python 3.9**
- **Pytorch 1.9** (it may work on some lower versions, but not tested)

Please use [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) to build the environment:
```shell
conda create -n dvf python=3.9
source activate dvf
```
### Install diffvg

diffvg is utilized to refine our generated vector glyphs in the testing phase.
Please go to https://github.com/BachiLi/diffvg see how to install it.

## Data and Pretrained-model

will be released soon...

## Training and Testing

to train our main model, run
```
python main.py --mode train --experiment_name dvf --model_name main_model
```

to train the neural rasterizer
python train_nr.py --mode train --experiment_name dvf --model_name neural_raster

```
CUDA_VISIBLE_DEVICES=0 python train_sr.py --mode train --experiment_name image_ss --model_name image_sr
```

to test our main model, run

python test_sf.py --mode test --experiment_name dvf --model_name main_model --test_epoch 625 --batch_size 1

to test our main model, run
```
python refinement.mp.py --experiment_name dvf --expid 4
```
where the `expid` denotes the index of testing font.