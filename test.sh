set -eux

pretrained_model=./pretrained_lm/roberta_base/
data_dir=./data/DocRED/
checkpoint_dir=./checkpoints/
predict_thresh=0.46544307

python ./pipeline.py \
  --model_type roberta \
  --entity_structure biaffine \
  --model_name_or_path ${pretrained_model} \
  --do_predict \
  --predict_thresh $predict_thresh \
  --data_dir ${data_dir} \
  --max_seq_length 512 \
  --max_ent_cnt 100 \
  --checkpoint_dir $checkpoint_dir \
  --seed 42
