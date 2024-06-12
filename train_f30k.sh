DATASET_NAME='f30k'
DATA_PATH='./data/'${DATASET_NAME}

CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} \
  --logger_name ./models/itmAFA_${DATASET_NAME}/log --model_name ./models/itmAFA_${DATASET_NAME} \
  --num_epochs=30 --lr_update=15 --learning_rate=.0005 --precomp_enc_type basic --workers 2 \
  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1
