MODEL_DIR="/home/zoro/work/YelpAnomalyAnalysis/Classify/Chainer/src_af_20151215/classification/models"
MODEL_FC=$MODEL_DIR".rv_classification_models@NetModel_FC"
python ../utils/training.py --task "rv_topics" --model $MODEL_FC --gpu -1 --training 2000 --norm 0 --index 0 --site yelp
