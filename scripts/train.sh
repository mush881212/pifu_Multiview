#python -m apps.train_shape --dataroot traindata --num_epoch 4 --batch_size 3 --num_views 4 --random_flip --random_scale --random_trans

# command
python ./apps/train_color.py \
    --batch_size 5 \
    --dataroot traindata\
    --hg_down 'ave_pool' \
    --norm 'group' \
    --norm_color 'group' \
    --num_views 4\
    --num_epoch 2\
    --num_sample_inout 0\
    --num_sample_color 5000\
    --sigma 0.1\
    --random_flip --random_scale --random_trans
