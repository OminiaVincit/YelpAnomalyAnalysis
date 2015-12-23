MODEL_DIR="/home/zoro/work/YelpAnomalyAnalysis/Classify/Chainer/src_af_20151215/classification/models"
MODEL_FC=$MODEL_DIR".rv_classification_models@NetModel_FC"
python ../utils/training.py --task "rv_check" --result-dir "old_divider" --model $MODEL_FC --gpu -1 --training 1000 --norm 1 --index 0 --site tripadvisor
python ../utils/training.py --task "rv_check" --result-dir "old_divider" --model $MODEL_FC --gpu -1 --training 1000 --norm 1 --index 2 --site tripadvisor
python ../utils/training.py --task "rv_check" --result-dir "old_divider" --model $MODEL_FC --gpu -1 --training 1000 --norm 1 --index 4 --site tripadvisor
python ../utils/training.py --task "rv_check" --result-dir "old_divider" --model $MODEL_FC --gpu -1 --training 1000 --norm 1 --index 6 --site tripadvisor
python ../utils/training.py --task "rv_check" --result-dir "old_divider" --model $MODEL_FC --gpu -1 --training 1000 --norm 1 --index 8 --site tripadvisor
