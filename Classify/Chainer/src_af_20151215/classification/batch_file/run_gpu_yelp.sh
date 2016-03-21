MODEL_DIR="/home/zoro/work/YelpAnomalyAnalysis/Classify/Chainer/src_af_20151215/classification/models"
MODEL_BN=$MODEL_DIR".model_trier@NetModel_BN"

python ../utils/training.py --task "rv_class" --result-dir "yelp_TOPICS_MATRIX_64" --model $MODEL_BN --gpu -1 --ftype "TOPICS_MATRIX_64" --training  100 --norm -1 --index 0 --site yelp
