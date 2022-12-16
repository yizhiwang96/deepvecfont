# DeepVecFont

This is the official Pytorch implementation of the paper:

Yizhi Wang and Zhouhui Lian. DeepVecFont: Synthesizing High-quality Vector Fonts via Dual-modality Learning. SIGGRAPH Asia. 2021.

Paper: [arxiv](https://arxiv.org/abs/2110.06688)
Supplementary: [Link](https://yizhiwang96.github.io/deepvecfont_homepage/pdfs/deepvecfont_sm.pdf)
Homepage: [DeepVecFont](https://yizhiwang96.github.io/deepvecfont_homepage/)

<div align=center>
	<img src="imgs/teaser.svg"> 
</div>

## Updates
### 2022-07-03
- DeepVecFont can now read data from single files when training, which significantly reduces memory consuming.
- DeepVecFont now better supports training images with resolution of 128 * 128, which should give better results. (the default image size now is 128 * 128, switch back to 64 by setting `image_size` in `options.py` to `64`.)
- [To Do] adding Chinese vector font support (will be available soon)
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

### Vector font interpolation
Rendered:
<div align=center>
	<img src="imgs/font_35_raw/syn/18.svg"> 
	<img src="imgs/font_35_raw/syn/08.svg"> 
	<img src="imgs/font_35_raw/syn/06.svg"> 
	<img src="imgs/font_35_raw/syn/06.svg">
	<img src="imgs/font_35_raw/syn/17.svg">
	<img src="imgs/font_35_raw/syn/00.svg">
	<img src="imgs/font_35_raw/syn/15.svg"> 
	<img src="imgs/font_35_raw/syn/07.svg">
	<img src="imgs/font_35_raw/syn/03.svg">
	<img src="imgs/font_35_raw/syn/30.svg">
	<img src="imgs/font_35_raw/syn/38.svg"> 
	<img src="imgs/font_35_raw/syn/40.svg">
	<br/>
	<img src="imgs/font_35_57_3_raw/syn/18.svg"> 
	<img src="imgs/font_35_57_3_raw/syn/08.svg"> 
	<img src="imgs/font_35_57_3_raw/syn/06.svg"> 
	<img src="imgs/font_35_57_3_raw/syn/06.svg">
	<img src="imgs/font_35_57_3_raw/syn/17.svg">
	<img src="imgs/font_35_57_3_raw/syn/00.svg">
	<img src="imgs/font_35_57_3_raw/syn/15.svg"> 
	<img src="imgs/font_35_57_3_raw/syn/07.svg"> 
	<img src="imgs/font_35_57_3_raw/syn/03.svg">
	<img src="imgs/font_35_57_3_raw/syn/30.svg">
	<img src="imgs/font_35_57_3_raw/syn/38.svg"> 
	<img src="imgs/font_35_57_3_raw/syn/40.svg">
	<br/>
	<img src="imgs/font_35_57_5_raw/syn/18.svg"> 
	<img src="imgs/font_35_57_5_raw/syn/08.svg"> 
	<img src="imgs/font_35_57_5_raw/syn/06.svg"> 
	<img src="imgs/font_35_57_5_raw/syn/06.svg">
	<img src="imgs/font_35_57_5_raw/syn/17.svg">
	<img src="imgs/font_35_57_5_raw/syn/00.svg">
	<img src="imgs/font_35_57_5_raw/syn/15.svg"> 
	<img src="imgs/font_35_57_5_raw/syn/07.svg"> 
	<img src="imgs/font_35_57_5_raw/syn/03.svg">
	<img src="imgs/font_35_57_5_raw/syn/30.svg">
	<img src="imgs/font_35_57_5_raw/syn/38.svg"> 
	<img src="imgs/font_35_57_5_raw/syn/40.svg">
	<br/>
	<img src="imgs/font_35_57_7_raw/syn/18.svg"> 
	<img src="imgs/font_35_57_7_raw/syn/08.svg"> 
	<img src="imgs/font_35_57_7_raw/syn/06.svg"> 
	<img src="imgs/font_35_57_7_raw/syn/06.svg">
	<img src="imgs/font_35_57_7_raw/syn/17.svg">
	<img src="imgs/font_35_57_7_raw/syn/00.svg">
	<img src="imgs/font_35_57_7_raw/syn/15.svg"> 
	<img src="imgs/font_35_57_7_raw/syn/07.svg">
	<img src="imgs/font_35_57_7_raw/syn/03.svg">
	<img src="imgs/font_35_57_7_raw/syn/30.svg">
	<img src="imgs/font_35_57_7_raw/syn/38.svg"> 
	<img src="imgs/font_35_57_7_raw/syn/40.svg">
	<br/>
	<img src="imgs/font_57_raw/syn/18.svg"> 
	<img src="imgs/font_57_raw/syn/08.svg"> 
	<img src="imgs/font_57_raw/syn/06.svg"> 
	<img src="imgs/font_57_raw/syn/06.svg">
	<img src="imgs/font_57_raw/syn/17.svg">
	<img src="imgs/font_57_raw/syn/00.svg">
	<img src="imgs/font_57_raw/syn/15.svg"> 
	<img src="imgs/font_57_raw/syn/07.svg">
	<img src="imgs/font_57_raw/syn/03.svg">
	<img src="imgs/font_57_raw/syn/30.svg">
	<img src="imgs/font_57_raw/syn/38.svg"> 
	<img src="imgs/font_57_raw/syn/40.svg">
	<br/>
</div>
Outlines:
<div align=center>
	<img src="imgs/font_35/syn/35_18.svg"> 
	<img src="imgs/font_35/syn/35_08.svg"> 
	<img src="imgs/font_35/syn/35_06.svg"> 
	<img src="imgs/font_35/syn/35_06.svg">
	<img src="imgs/font_35/syn/35_17.svg">
	<img src="imgs/font_35/syn/35_00.svg">
	<img src="imgs/font_35/syn/35_15.svg"> 
	<img src="imgs/font_35/syn/35_07.svg">
	<img src="imgs/font_35/syn/35_03.svg">
	<img src="imgs/font_35/syn/35_30.svg">
	<img src="imgs/font_35/syn/35_38.svg"> 
	<img src="imgs/font_35/syn/35_40.svg">
	<br/>
	<img src="imgs/font_35_57_3/syn/35_57_3_18.svg"> 
	<img src="imgs/font_35_57_3/syn/35_57_3_08.svg"> 
	<img src="imgs/font_35_57_3/syn/35_57_3_06.svg"> 
	<img src="imgs/font_35_57_3/syn/35_57_3_06.svg">
	<img src="imgs/font_35_57_3/syn/35_57_3_17.svg">
	<img src="imgs/font_35_57_3/syn/35_57_3_00.svg">
	<img src="imgs/font_35_57_3/syn/35_57_3_15.svg"> 
	<img src="imgs/font_35_57_3/syn/35_57_3_07.svg"> 
	<img src="imgs/font_35_57_3/syn/35_57_3_03.svg">
	<img src="imgs/font_35_57_3/syn/35_57_3_30.svg">
	<img src="imgs/font_35_57_3/syn/35_57_3_38.svg"> 
	<img src="imgs/font_35_57_3/syn/35_57_3_40.svg">	
	<br/>
	<img src="imgs/font_35_57_5/syn/35_57_5_18.svg"> 
	<img src="imgs/font_35_57_5/syn/35_57_5_08.svg"> 
	<img src="imgs/font_35_57_5/syn/35_57_5_06.svg"> 
	<img src="imgs/font_35_57_5/syn/35_57_5_06.svg">
	<img src="imgs/font_35_57_5/syn/35_57_5_17.svg">
	<img src="imgs/font_35_57_5/syn/35_57_5_00.svg">
	<img src="imgs/font_35_57_5/syn/35_57_5_15.svg"> 
	<img src="imgs/font_35_57_5/syn/35_57_5_07.svg"> 
	<img src="imgs/font_35_57_5/syn/35_57_5_03.svg">
	<img src="imgs/font_35_57_5/syn/35_57_5_30.svg">
	<img src="imgs/font_35_57_5/syn/35_57_5_38.svg"> 
	<img src="imgs/font_35_57_5/syn/35_57_5_40.svg">	
	<br/>
	<img src="imgs/font_35_57_7/syn/35_57_7_18.svg"> 
	<img src="imgs/font_35_57_7/syn/35_57_7_08.svg"> 
	<img src="imgs/font_35_57_7/syn/35_57_7_06.svg"> 
	<img src="imgs/font_35_57_7/syn/35_57_7_06.svg">
	<img src="imgs/font_35_57_7/syn/35_57_7_17.svg">
	<img src="imgs/font_35_57_7/syn/35_57_7_00.svg">
	<img src="imgs/font_35_57_7/syn/35_57_7_15.svg"> 
	<img src="imgs/font_35_57_7/syn/35_57_7_07.svg">
	<img src="imgs/font_35_57_7/syn/35_57_7_03.svg">
	<img src="imgs/font_35_57_7/syn/35_57_7_30.svg">
	<img src="imgs/font_35_57_7/syn/35_57_7_38.svg"> 
	<img src="imgs/font_35_57_7/syn/35_57_7_40.svg">
	<br/>
	<img src="imgs/font_57/syn/57_18.svg"> 
	<img src="imgs/font_57/syn/57_08.svg"> 
	<img src="imgs/font_57/syn/57_06.svg"> 
	<img src="imgs/font_57/syn/57_06.svg">
	<img src="imgs/font_57/syn/57_17.svg">
	<img src="imgs/font_57/syn/57_00.svg">
	<img src="imgs/font_57/syn/57_15.svg"> 
	<img src="imgs/font_57/syn/57_07.svg"> 
	<img src="imgs/font_57/syn/57_03.svg">
	<img src="imgs/font_57/syn/57_30.svg">
	<img src="imgs/font_57/syn/57_38.svg"> 
	<img src="imgs/font_57/syn/57_40.svg">
	<br/>
</div>

### Random generation (New fonts)
Rendered:
<div align=center>
	<img src="imgs/font_18r_raw/syn/00.svg"> 
	<img src="imgs/font_18r_raw/syn/01.svg"> 
	<img src="imgs/font_18r_raw/syn/02.svg"> 
	<img src="imgs/font_18r_raw/syn/03.svg"> 
	<img src="imgs/font_18r_raw/syn/04.svg">
	<img src="imgs/font_18r_raw/syn/05.svg">
	<img src="imgs/font_18r_raw/syn/06.svg">
	<img src="imgs/font_18r_raw/syn/07.svg"> 
	<img src="imgs/font_18r_raw/syn/08.svg"> 
	<img src="imgs/font_18r_raw/syn/09.svg"> 
	<img src="imgs/font_18r_raw/syn/10.svg"> 
	<img src="imgs/font_18r_raw/syn/11.svg">
	<img src="imgs/font_18r_raw/syn/12.svg">
	<img src="imgs/font_18r_raw/syn/13.svg">
	<img src="imgs/font_18r_raw/syn/14.svg"> 
	<img src="imgs/font_18r_raw/syn/15.svg"> 
	<img src="imgs/font_18r_raw/syn/16.svg"> 
	<img src="imgs/font_18r_raw/syn/17.svg"> 
	<img src="imgs/font_18r_raw/syn/18.svg">
	<img src="imgs/font_18r_raw/syn/19.svg">
	<img src="imgs/font_18r_raw/syn/20.svg">
	<img src="imgs/font_18r_raw/syn/21.svg"> 
	<img src="imgs/font_18r_raw/syn/22.svg"> 
	<img src="imgs/font_18r_raw/syn/23.svg"> 	
	<img src="imgs/font_18r_raw/syn/24.svg">
	<img src="imgs/font_18r_raw/syn/25.svg">
	<br/>
	<img src="imgs/font_18r_raw/syn/26.svg"> 
	<img src="imgs/font_18r_raw/syn/27.svg"> 
	<img src="imgs/font_18r_raw/syn/28.svg"> 
	<img src="imgs/font_18r_raw/syn/29.svg"> 
	<img src="imgs/font_18r_raw/syn/30.svg">
	<img src="imgs/font_18r_raw/syn/31.svg">
	<img src="imgs/font_18r_raw/syn/32.svg">
	<img src="imgs/font_18r_raw/syn/33.svg"> 
	<img src="imgs/font_18r_raw/syn/34.svg"> 
	<img src="imgs/font_18r_raw/syn/35.svg"> 
	<img src="imgs/font_18r_raw/syn/36.svg"> 
	<img src="imgs/font_18r_raw/syn/37.svg">
	<img src="imgs/font_18r_raw/syn/38.svg">
	<img src="imgs/font_18r_raw/syn/39.svg">
	<img src="imgs/font_18r_raw/syn/40.svg"> 
	<img src="imgs/font_18r_raw/syn/41.svg"> 
	<img src="imgs/font_18r_raw/syn/42.svg"> 
	<img src="imgs/font_18r_raw/syn/43.svg"> 
	<img src="imgs/font_18r_raw/syn/44.svg">
	<img src="imgs/font_18r_raw/syn/45.svg">
	<img src="imgs/font_18r_raw/syn/46.svg">
	<img src="imgs/font_18r_raw/syn/47.svg"> 
	<img src="imgs/font_18r_raw/syn/48.svg"> 
	<img src="imgs/font_18r_raw/syn/49.svg">
	<img src="imgs/font_18r_raw/syn/50.svg">
	<img src="imgs/font_18r_raw/syn/51.svg">	
	<br/>
</div>

## Installation

### Requirement

- **python 3.9**
- **Pytorch 1.9** (it may work on some lower versions, but not tested; the contextual loss can't work in Pytorch 1.10)

Please use [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) to build the environment:
```shell
conda create -n dvf python=3.9
source activate dvf
```
Install pytorch via the [instructions](https://pytorch.org/get-started/locally/).
- Others
```shell
conda install tensorboardX scikit-image
```

### Install diffvg

We utilize diffvg to refine our generated vector glyphs in the testing phase.
Please go to https://github.com/BachiLi/diffvg see how to install it.

**Important (updated 2021.10.19):** You need first replace the original `diffvg/pydiffvg/save_svg.py` with [this](./data_utils/save_svg.py) and then install.

## Data and Pretrained-model

### Dataset
#### **The Vector Font dataset**
(This dataset is a subset from [SVG-VAE](https://github.com/magenta/magenta/tree/main/magenta/models/svg_vae), ICCV 2019.)
There are two modes to access the dataset:
- **pkl files:** (Download links: [Google Drive](https://drive.google.com/drive/folders/1dGOOXK63-QJKXnE7_fD2OCfYJGKsApSg?usp=sharing))
Using pkl files (that combine all fonts) consumes a lot of memory, but you don't need to generate a large number of single files.
Please download the `vecfont_dataset_pkls` dir and put it under `./data/`.
If you use this mode, set `read_mode` = `pkl`, in `options.py`. If you use the uploaded data, set `image_size` to `64`.
- **dirs:**
Generate directories for each data entry, which consumes much less memory.
You need to first download the ttf/otf files from [Google Drive](https://drive.google.com/file/d/1D-KxfDqpz1tOSY5VxfsU7o0HlKd0GuJI/view?usp=sharing) and extract it to `data_utils`.
Then follow the instructions in `Customize your own dataset` to process these ttf files. The dataset will be saved with the name of `vecfont_dataset_dirs`.
If you use this mode, set `read_mode` = `dirs`, in `options.py`.

#### **The Image Super-resolution dataset** 

This dataset is too huge so I suggest to create it by yourself (see the details below about creating your own dataset).

### Pretrained models
#### **The Neural Rasterizer** 
Download link: [image size 64x64](https://drive.google.com/drive/folders/10Qy7vFn27H2qQfve1Tu7UR3sm3l45cKg?usp=sharing), [image size 128x128](https://drive.google.com/drive/folders/1cKobKq74yDv9UGCoZ1WjHgjBN7sF3puS?usp=sharing). Please download the `dvf_neural_raster` dir and put it under `./experiments/`.

#### **The Image Super-resolution model**
Download link: [image size 64x64 -> 256x256](https://drive.google.com/drive/folders/1D_U4KHbt42u6ZGNNOAOvy5QXjwHj_abX?usp=sharing), [image size 128x128 -> 256x256](https://drive.google.com/drive/folders/157WTmx3o95K2c9KR-1Bznfl0J0EPqKOv?usp=sharing). Please download the `image_sr` dir and put it under `./experiments/`.

#### **The Main model** 
Download link: [image size 64x64](https://drive.google.com/drive/folders/1Ykacqk8VPhvVv_XFH47BBvGt8ZlBrpnZ?usp=sharing), [image size 128x128](https://drive.google.com/drive/folders/120B0oTXjsIgo8a-pyg1ZL_ZyC2eE3xNq?usp=sharing). Please download the `dvf_main_model` dir and put it under `./experiments/`.

Note that recently we switched from Tensorflow to Pytorch, we may update the models that have better performances.

## Training and Testing

To train our main model, run
```
python main.py --mode train --experiment_name dvf --model_name main_model
```
The configurations can be found in `options.py`.

To test our main model, run
```
python test_sf.py --mode test --experiment_name dvf --model_name main_model --test_epoch 1200 --batch_size 1 --mix_temperature 0.0001 --gauss_temperature 0.01
```
This will output the synthesized fonts without refinements. Note that `batch_size` must be set to 1. The results will be written in `./experiments/dvf_main_model/results/`.


To refinement the vector glyphs, run
```
python refinement_mp.py --experiment_name dvf --fontid 14 --candidate_nums 20 --num_processes 4
```
where the `fontid` denotes the index of testing font. The results will be written in `./experiments/dvf_main_model/results/0014/svgs_refined/`. Set `num_processes` according to your GPU's computation capacity. Setting `init_svgbbox_align2img` to `True` could give better results when the initial svg and raster image don't align well.

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

## Customize your own dataset

- **Prepare ttf/otf files**

Put the ttf/otf files in `./data_utils/font_ttfs/train` and `./data_utils/font_ttfs/test`, and organize them as `0000.ttf`, `0001.ttf`, `0002.ttf`...
The ttf/otf files in our dataset can be found in [Google Drive](https://drive.google.com/file/d/1D-KxfDqpz1tOSY5VxfsU7o0HlKd0GuJI/view?usp=sharing).

- **Deactivate the conda environment and install Fontforge**

for python > 3.0:
```
conda deactivate
apt install python3-fontforge
```
It works in Ubuntu 20.04.1, other lower versions may fail in `import fontforge`.
- **Get SFD files via Fontforge**
```
cd data_utils
python convert_ttf_to_sfd_mp.py --split train
python convert_ttf_to_sfd_mp.py --split test
```

- **Generate glyph images**
```
python write_glyph_imgs.py --split train
python write_glyph_imgs.py --split test
```

- **package them to dirs or pkl**

dirs (recommended):
```
python write_data_to_dirs.py --split train
python write_data_to_dirs.py --split test
```

pkl
```
python write_data_to_pkl.py --split train
python write_data_to_pkl.py --split test
```
Note:

(1) If you use the mean and stddev files calculated from your own data, you need to retrain the neural rasterizer. For English fonts, just use the mean and stddev files we provided.

## Acknowledgment

- [SVG-VAE](https://github.com/magenta/magenta/tree/main/magenta/models/svg_vae)
- [SVG-VAE-pytorch](https://github.com/hologerry/svg_vae_pytorch) by [hologerry](https://github.com/hologerry)
- [Diffvg](https://github.com/BachiLi/diffvg)

## Citation

If you use this code or find our work is helpful, please consider citing our work:
```
@article{wang2021deepvecfont,
 author = {Wang, Yizhi and Lian, Zhouhui},
 title = {DeepVecFont: Synthesizing High-quality Vector Fonts via Dual-modality Learning},
 journal = {ACM Transactions on Graphics},
 numpages = {15},
 volume={40},
 number={6},
 month = December,
 year = {2021},
 doi={10.1145/3478513.3480488}
}
```
