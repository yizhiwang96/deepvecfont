# The official implemention of DeepVecFont, SIGGRAPH Asia 2021

# install



# run



CUDA_VISIBLE_DEVICES=0 python train_sr.py --mode train --experiment_name image_ss --model_name image_sr

CUDA_VISIBLE_DEVICES=0 python train_nr.py --mode train --experiment_name datarck_fulldata --model_name neural_raster

CUDA_VISIBLE_DEVICES=0 python main.py --mode train --experiment_name contextual_loss_noseqinput --model_name main_model

CUDA_VISIBLE_DEVICES=0 python test.py --mode test --experiment_name gumbel_main_model --model_name main_model --test_epoch 175

CUDA_VISIBLE_DEVICES=0 python test_sf.py --mode test --experiment_name argmax_v1.0 --model_name main_model --test_epoch 625 --batch_size 1