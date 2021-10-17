# DeepVecFont

This is the official Pytorch implementation of the paper:

Yizhi Wang and Zhouhui Lian. DeepVecFont: Synthesizing High-quality Vector Fonts via Dual-modality Learning. SIGGRAPH Asia. 2021.

Paper: [arxiv](https://arxiv.org/abs/2110.06688)

<div align=center>
	<img src="imgs/teaser.svg"> 
</div>

## Demo
### Few-shot generation
Given a few vector glyphs of a font as reference, our model generates the full **vector** font:

Input glyphs:
<div align=center>
	<img src="imgs/font_02/gt/02_00.svg"> 
	<img src="imgs/font_02/gt/02_01.svg"> 
	<img src="imgs/font_02/gt/02_26.svg"> 
	<img src="imgs/font_02/gt/02_27.svg"> 
</div>
Synthesized glyphs by DeepVecFont:
<div align=center>
	<img src="imgs/font_02/syn/02_00.svg"> 
	<img src="imgs/font_02/syn/02_01.svg"> 
	<img src="imgs/font_02/syn/02_02.svg"> 
	<img src="imgs/font_02/syn/02_03.svg"> 
	<img src="imgs/font_02/syn/02_04.svg">
	<img src="imgs/font_02/syn/02_05.svg">
	<img src="imgs/font_02/syn/02_06.svg">
	<img src="imgs/font_02/syn/02_07.svg"> 
	<img src="imgs/font_02/syn/02_08.svg"> 
	<img src="imgs/font_02/syn/02_09.svg"> 
	<img src="imgs/font_02/syn/02_10.svg"> 
	<img src="imgs/font_02/syn/02_11.svg">
	<img src="imgs/font_02/syn/02_12.svg">
	<img src="imgs/font_02/syn/02_13.svg">
	<img src="imgs/font_02/syn/02_14.svg"> 
	<img src="imgs/font_02/syn/02_15.svg"> 
	<img src="imgs/font_02/syn/02_16.svg"> 
	<img src="imgs/font_02/syn/02_17.svg"> 
	<img src="imgs/font_02/syn/02_18.svg">
	<img src="imgs/font_02/syn/02_19.svg">
	<img src="imgs/font_02/syn/02_20.svg">
	<img src="imgs/font_02/syn/02_21.svg"> 
	<img src="imgs/font_02/syn/02_22.svg"> 
	<img src="imgs/font_02/syn/02_23.svg"> 	
	<img src="imgs/font_02/syn/02_24.svg">
	<img src="imgs/font_02/syn/02_25.svg">
	<br/>
	<img src="imgs/font_02/syn/02_26.svg"> 
	<img src="imgs/font_02/syn/02_27.svg"> 
	<img src="imgs/font_02/syn/02_28.svg"> 
	<img src="imgs/font_02/syn/02_29.svg"> 
	<img src="imgs/font_02/syn/02_30.svg">
	<img src="imgs/font_02/syn/02_31.svg">
	<img src="imgs/font_02/syn/02_32.svg">
	<img src="imgs/font_02/syn/02_33.svg"> 
	<img src="imgs/font_02/syn/02_34.svg"> 
	<img src="imgs/font_02/syn/02_35.svg"> 
	<img src="imgs/font_02/syn/02_36.svg"> 
	<img src="imgs/font_02/syn/02_37.svg">
	<img src="imgs/font_02/syn/02_38.svg">
	<img src="imgs/font_02/syn/02_39.svg">
	<img src="imgs/font_02/syn/02_40.svg"> 
	<img src="imgs/font_02/syn/02_41.svg"> 
	<img src="imgs/font_02/syn/02_42.svg"> 
	<img src="imgs/font_02/syn/02_43.svg"> 
	<img src="imgs/font_02/syn/02_44.svg">
	<img src="imgs/font_02/syn/02_45.svg">
	<img src="imgs/font_02/syn/02_46.svg">
	<img src="imgs/font_02/syn/02_47.svg"> 
	<img src="imgs/font_02/syn/02_48.svg"> 
	<img src="imgs/font_02/syn/02_49.svg">
	<img src="imgs/font_02/syn/02_50.svg">
	<img src="imgs/font_02/syn/02_51.svg">	
	<br/>
</div>

Input glyphs:
<div align=center>
	<img src="imgs/font_12/gt/12_00.svg"> 
	<img src="imgs/font_12/gt/12_01.svg"> 
	<img src="imgs/font_12/gt/12_26.svg"> 
	<img src="imgs/font_12/gt/12_27.svg"> 
</div>
Synthesized glyphs by DeepVecFont:
<div align=center>
	<img src="imgs/font_12/syn/12_00.svg"> 
	<img src="imgs/font_12/syn/12_01.svg"> 
	<img src="imgs/font_12/syn/12_02.svg"> 
	<img src="imgs/font_12/syn/12_03.svg"> 
	<img src="imgs/font_12/syn/12_04.svg">
	<img src="imgs/font_12/syn/12_05.svg">
	<img src="imgs/font_12/syn/12_06.svg">
	<img src="imgs/font_12/syn/12_07.svg"> 
	<img src="imgs/font_12/syn/12_08.svg"> 
	<img src="imgs/font_12/syn/12_09.svg"> 
	<img src="imgs/font_12/syn/12_10.svg"> 
	<img src="imgs/font_12/syn/12_11.svg">
	<img src="imgs/font_12/syn/12_12.svg">
	<img src="imgs/font_12/syn/12_13.svg">
	<img src="imgs/font_12/syn/12_14.svg"> 
	<img src="imgs/font_12/syn/12_15.svg"> 
	<img src="imgs/font_12/syn/12_16.svg"> 
	<img src="imgs/font_12/syn/12_17.svg"> 
	<img src="imgs/font_12/syn/12_18.svg">
	<img src="imgs/font_12/syn/12_19.svg">
	<img src="imgs/font_12/syn/12_20.svg">
	<img src="imgs/font_12/syn/12_21.svg"> 
	<img src="imgs/font_12/syn/12_22.svg"> 
	<img src="imgs/font_12/syn/12_23.svg"> 	
	<img src="imgs/font_12/syn/12_24.svg">
	<img src="imgs/font_12/syn/12_25.svg">
	<br/>
	<img src="imgs/font_12/syn/12_26.svg"> 
	<img src="imgs/font_12/syn/12_27.svg"> 
	<img src="imgs/font_12/syn/12_28.svg"> 
	<img src="imgs/font_12/syn/12_29.svg"> 
	<img src="imgs/font_12/syn/12_30.svg">
	<img src="imgs/font_12/syn/12_31.svg">
	<img src="imgs/font_12/syn/12_32.svg">
	<img src="imgs/font_12/syn/12_33.svg"> 
	<img src="imgs/font_12/syn/12_34.svg"> 
	<img src="imgs/font_12/syn/12_35.svg"> 
	<img src="imgs/font_12/syn/12_36.svg"> 
	<img src="imgs/font_12/syn/12_37.svg">
	<img src="imgs/font_12/syn/12_38.svg">
	<img src="imgs/font_12/syn/12_39.svg">
	<img src="imgs/font_12/syn/12_40.svg"> 
	<img src="imgs/font_12/syn/12_41.svg"> 
	<img src="imgs/font_12/syn/12_42.svg"> 
	<img src="imgs/font_12/syn/12_43.svg"> 
	<img src="imgs/font_12/syn/12_44.svg">
	<img src="imgs/font_12/syn/12_45.svg">
	<img src="imgs/font_12/syn/12_46.svg">
	<img src="imgs/font_12/syn/12_47.svg"> 
	<img src="imgs/font_12/syn/12_48.svg"> 
	<img src="imgs/font_12/syn/12_49.svg">
	<img src="imgs/font_12/syn/12_50.svg">
	<img src="imgs/font_12/syn/12_51.svg">	
	<br/>
</div>

Input glyphs:
<div align=center>
	<img src="imgs/font_41/gt/41_00.svg"> 
	<img src="imgs/font_41/gt/41_01.svg"> 
	<img src="imgs/font_41/gt/41_26.svg"> 
	<img src="imgs/font_41/gt/41_27.svg"> 
</div>
Synthesized glyphs by DeepVecFont:
<div align=center>
	<img src="imgs/font_41/syn/41_00.svg"> 
	<img src="imgs/font_41/syn/41_01.svg"> 
	<img src="imgs/font_41/syn/41_02.svg"> 
	<img src="imgs/font_41/syn/41_03.svg"> 
	<img src="imgs/font_41/syn/41_04.svg">
	<img src="imgs/font_41/syn/41_05.svg">
	<img src="imgs/font_41/syn/41_06.svg">
	<img src="imgs/font_41/syn/41_07.svg"> 
	<img src="imgs/font_41/syn/41_08.svg"> 
	<img src="imgs/font_41/syn/41_09.svg"> 
	<img src="imgs/font_41/syn/41_10.svg"> 
	<img src="imgs/font_41/syn/41_11.svg">
	<img src="imgs/font_41/syn/41_12.svg">
	<img src="imgs/font_41/syn/41_13.svg">
	<img src="imgs/font_41/syn/41_14.svg"> 
	<img src="imgs/font_41/syn/41_15.svg"> 
	<img src="imgs/font_41/syn/41_16.svg"> 
	<img src="imgs/font_41/syn/41_17.svg"> 
	<img src="imgs/font_41/syn/41_18.svg">
	<img src="imgs/font_41/syn/41_19.svg">
	<img src="imgs/font_41/syn/41_20.svg">
	<img src="imgs/font_41/syn/41_21.svg"> 
	<img src="imgs/font_41/syn/41_22.svg"> 
	<img src="imgs/font_41/syn/41_23.svg"> 	
	<img src="imgs/font_41/syn/41_24.svg">
	<img src="imgs/font_41/syn/41_25.svg">
	<br/>
	<img src="imgs/font_41/syn/41_26.svg"> 
	<img src="imgs/font_41/syn/41_27.svg"> 
	<img src="imgs/font_41/syn/41_28.svg"> 
	<img src="imgs/font_41/syn/41_29.svg"> 
	<img src="imgs/font_41/syn/41_30.svg">
	<img src="imgs/font_41/syn/41_31.svg">
	<img src="imgs/font_41/syn/41_32.svg">
	<img src="imgs/font_41/syn/41_33.svg"> 
	<img src="imgs/font_41/syn/41_34.svg"> 
	<img src="imgs/font_41/syn/41_35.svg"> 
	<img src="imgs/font_41/syn/41_36.svg"> 
	<img src="imgs/font_41/syn/41_37.svg">
	<img src="imgs/font_41/syn/41_38.svg">
	<img src="imgs/font_41/syn/41_39.svg">
	<img src="imgs/font_41/syn/41_40.svg"> 
	<img src="imgs/font_41/syn/41_41.svg"> 
	<img src="imgs/font_41/syn/41_42.svg"> 
	<img src="imgs/font_41/syn/41_43.svg"> 
	<img src="imgs/font_41/syn/41_44.svg">
	<img src="imgs/font_41/syn/41_45.svg">
	<img src="imgs/font_41/syn/41_46.svg">
	<img src="imgs/font_41/syn/41_47.svg"> 
	<img src="imgs/font_41/syn/41_48.svg"> 
	<img src="imgs/font_41/syn/41_49.svg">
	<img src="imgs/font_41/syn/41_50.svg">
	<img src="imgs/font_41/syn/41_51.svg">	
	<br/>
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
Install pytorch via the [instructions](https://pytorch.org/get-started/locally/).

### Install diffvg

We utilize diffvg to refine our generated vector glyphs in the testing phase.
Please go to https://github.com/BachiLi/diffvg see how to install it.

## Data and Pretrained-model

### Dataset
- **The Vector Font dataset** Download links: [Google Drive](https://drive.google.com/drive/folders/1dGOOXK63-QJKXnE7_fD2OCfYJGKsApSg?usp=sharing)

Please download the `vecfont_dataset` dir and put it under `./data/`.
(This dataset is a subset from [SVG-VAE](https://github.com/magenta/magenta/tree/main/magenta/models/svg_vae), ICCV 2019.
We will release more information about how to create from your own data.)

- **The mean and stdev files** Download links: [Google Drive](https://drive.google.com/drive/folders/1ZDZQIf2LGXmlKKPtkS3l32P77iNAlBqD?usp=sharing)

Please Download them and put it under `./data/`.

### Pretrained model
- **The Neural Rasterizer** Download links: [Google Drive](https://drive.google.com/drive/folders/10Qy7vFn27H2qQfve1Tu7UR3sm3l45cKg?usp=sharing)

Please download the `dvf_neural_raster` dir and put it under `./experiments/`.

- **The Image Super-resolution model**  Download links: [Google Drive](https://drive.google.com/drive/folders/1D_U4KHbt42u6ZGNNOAOvy5QXjwHj_abX?usp=sharing).

Please download the `image_sr` dir and put it under `./experiments/`.
Note that recently we switched from Tensorflow to Pytorch, we may update the models that have better performances.

- **The Main model** Download links: [will be uploaded soon].

## Training and Testing

To train our main model, run
```
python main.py --mode train --experiment_name dvf --model_name main_model
```
The configurations can be found in `options.py`.

To test our main model, run
```
python test_sf.py --mode test --experiment_name dvf --model_name main_model --test_epoch 1500 --batch_size 1 --mix_temperature 0.0001 --gauss_temperature 0.01
```
This will output the synthesized fonts without refinements. Note that `batch_size` must be set to 1. The results will be written in `./experiments/dvf/`.


To refinement the vector glyphs, run
```
python refinement.mp.py --experiment_name dvf --fontid 14 --candidate_nums 20 
```
where the `fontid` denotes the index of testing font. The results will be written in `./experiments/dvf/results/0014/svgs_refined/`.

We have pretrained the neural rasterizer and image super-resolution model.
If you want to train them yourself:

To train the neural rasterizer:
```
python train_nr.py --mode train --experiment_name dvf --model_name neural_raster
```
To train the image super-resolution model:
```
python train_sr.py --mode train --name image_sr
```