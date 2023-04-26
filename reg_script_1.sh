# 1 -------------------------------------------------------------------------------
python set_seg_reg_config.py \
    --config_path configs/seg_reg.yaml \
    --output_path temp_configs/seg_reg/LPBA40_14/task1/seg_reg1.yaml \
    --train_data_path datasets/json/LPBA40_14.json \
    --train_gpu 1 \
    --epoch 1500 \
    --seg_step_per_epoch 1 \
    --reg_step_per_epoch 29 \
    --test_data_path datasets/json/LPBA40_14.json \
    --seg_lr 0.001 \
    --seg_step_size 9000 \
    --reg_lr 0.0002 \
    --reg_step_size 20000 \
    --gradient_loss_weight 1 \
    --supervised_loss_weight 1 \
    --anatomy_loss_weight 1

python main_seg_reg.py \
    --train \
    --config temp_configs/seg_reg/LPBA40_14/task1/seg_reg1.yaml \
    --output output/seg_reg/LPBA40_14/task1/train1

# 2 -------------------------------------------------------------------------------
python set_seg_reg_config.py \
    --config_path configs/seg_reg.yaml \
    --output_path temp_configs/seg_reg/LPBA40_14/task1/seg_reg2.yaml \
    --train_data_path datasets/json/LPBA40_14.json \
    --train_gpu 1 \
    --epoch 1500 \
    --seg_step_per_epoch 1 \
    --reg_step_per_epoch 29 \
    --test_data_path datasets/json/LPBA40_14.json \
    --seg_lr 0.001 \
    --seg_step_size 9000 \
    --reg_lr 0.0002 \
    --reg_step_size 20000 \
    --gradient_loss_weight 1 \
    --supervised_loss_weight 10 \
    --anatomy_loss_weight 1

python main_seg_reg.py \
    --train \
    --config temp_configs/seg_reg/LPBA40_14/task1/seg_reg2.yaml \
    --output output/seg_reg/LPBA40_14/task1/train2


# 3 -------------------------------------------------------------------------------
python set_seg_reg_config.py \
    --config_path configs/seg_reg.yaml \
    --output_path temp_configs/seg_reg/LPBA40_14/task1/seg_reg3.yaml \
    --train_data_path datasets/json/LPBA40_14.json \
    --train_gpu 1 \
    --epoch 1500 \
    --seg_step_per_epoch 1 \
    --reg_step_per_epoch 29 \
    --test_data_path datasets/json/LPBA40_14.json \
    --seg_lr 0.001 \
    --seg_step_size 9000 \
    --reg_lr 0.0002 \
    --reg_step_size 20000 \
    --gradient_loss_weight 1 \
    --supervised_loss_weight 100 \
    --anatomy_loss_weight 1

python main_seg_reg.py \
    --train \
    --config temp_configs/seg_reg/LPBA40_14/task1/seg_reg3.yaml \
    --output output/seg_reg/LPBA40_14/task1/train3

# 4 -------------------------------------------------------------------------------
python set_seg_reg_config.py \
    --config_path configs/seg_reg.yaml \
    --output_path temp_configs/seg_reg/LPBA40_14/task1/seg_reg4.yaml \
    --train_data_path datasets/json/LPBA40_14.json \
    --train_gpu 1 \
    --epoch 1500 \
    --seg_step_per_epoch 1 \
    --reg_step_per_epoch 29 \
    --test_data_path datasets/json/LPBA40_14.json \
    --seg_lr 0.001 \
    --seg_step_size 9000 \
    --reg_lr 0.0002 \
    --reg_step_size 20000 \
    --gradient_loss_weight 10 \
    --supervised_loss_weight 10 \
    --anatomy_loss_weight 1

python main_seg_reg.py \
    --train \
    --config temp_configs/seg_reg/LPBA40_14/task1/seg_reg4.yaml \
    --output output/seg_reg/LPBA40_14/task1/train4