MODEL_DIR="/home/zoro/work/YelpAnomalyAnalysis/Classify/Chainer/src_af_20151215/classification/models"
MODEL_CNN=$MODEL_DIR".rv_classification_models@NetModel_BN"
python ../utils/training.py --task "rv_topics" --model $MODEL_CNN --gpu -1 --training 2000 --norm 1 --index 0 --site yelp
