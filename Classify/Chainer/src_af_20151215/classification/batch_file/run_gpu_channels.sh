MODEL_DIR="/home/zoro/work/YelpAnomalyAnalysis/Classify/Chainer/src_af_20151215/classification/models"
MODEL_FC=$MODEL_DIR".rv_classification_models@NetModel_FC_nodrop_256_512"
#MODEL_BN=$MODEL_DIR".tfidf_models@NetModel_BN"

MODEL_TFIDF=$MODEL_DIR".tfidf_models@NetModel_FC_tfidf"
MODEL_TOPICS=$MODEL_DIR".tfidf_models@NetModel_FC_topics"
MODEL_LIWC=$MODEL_DIR".tfidf_models@NetModel_FC_LIWC"
MODEL_INQUIRER=$MODEL_DIR".tfidf_models@NetModel_FC_INQUIRER"
MODEL_GALC=$MODEL_DIR".tfidf_models@NetModel_FC_GALC"
MODEL_STR=$MODEL_DIR".tfidf_models@NetModel_FC_STR"

python ../utils/training.py --task "rv_class" --result-dir "yelp_TOPICS_64" --model $MODEL_TOPICS --gpu 0 --ftype "TOPICS_64" --training  200 --norm -1 --index 0 --site yelp
python ../utils/training.py --task "rv_class" --result-dir "yelp_TOPICS_64" --model $MODEL_TOPICS --gpu 0 --ftype "TOPICS_64" --training  200 --norm -1 --index 1 --site yelp
python ../utils/training.py --task "rv_class" --result-dir "yelp_TOPICS_64" --model $MODEL_TOPICS --gpu 0 --ftype "TOPICS_64" --training  200 --norm -1 --index 2 --site yelp

python ../utils/training.py --task "rv_class" --result-dir "yelp_STR" --model $MODEL_STR --gpu 0 --ftype "STR" --training  200 --norm -1 --index 0 --site yelp
python ../utils/training.py --task "rv_class" --result-dir "yelp_STR" --model $MODEL_STR --gpu 0 --ftype "STR" --training  200 --norm -1 --index 1 --site yelp
python ../utils/training.py --task "rv_class" --result-dir "yelp_STR" --model $MODEL_STR --gpu 0 --ftype "STR" --training  200 --norm -1 --index 2 --site yelp

python ../utils/training.py --task "rv_class" --result-dir "yelp_tfidf" --model $MODEL_TFIDF --gpu 0 --ftype "tfidf" --training  200 --norm -1 --index 0 --site yelp
python ../utils/training.py --task "rv_class" --result-dir "yelp_tfidf" --model $MODEL_TFIDF --gpu 0 --ftype "tfidf" --training  200 --norm -1 --index 1 --site yelp
python ../utils/training.py --task "rv_class" --result-dir "yelp_tfidf" --model $MODEL_TFIDF --gpu 0 --ftype "tfidf" --training  200 --norm -1 --index 2 --site yelp

python ../utils/training.py --task "rv_class" --result-dir "yelp_LIWC" --model $MODEL_LIWC --gpu 0 --ftype "LIWC" --training  200 --norm -1 --index 0 --site yelp
python ../utils/training.py --task "rv_class" --result-dir "yelp_LIWC" --model $MODEL_LIWC --gpu 0 --ftype "LIWC" --training  200 --norm -1 --index 1 --site yelp
python ../utils/training.py --task "rv_class" --result-dir "yelp_LIWC" --model $MODEL_LIWC --gpu 0 --ftype "LIWC" --training  200 --norm -1 --index 2 --site yelp

python ../utils/training.py --task "rv_class" --result-dir "yelp_INQUIRER" --model $MODEL_INQUIRER --gpu 0 --ftype "INQUIRER" --training  200 --norm -1 --index 0 --site yelp
python ../utils/training.py --task "rv_class" --result-dir "yelp_INQUIRER" --model $MODEL_INQUIRER --gpu 0 --ftype "INQUIRER" --training  200 --norm -1 --index 1 --site yelp
python ../utils/training.py --task "rv_class" --result-dir "yelp_INQUIRER" --model $MODEL_INQUIRER --gpu 0 --ftype "INQUIRER" --training  200 --norm -1 --index 2 --site yelp

python ../utils/training.py --task "rv_class" --result-dir "yelp_GALC" --model $MODEL_GALC --gpu 0 --ftype "GALC" --training  200 --norm -1 --index 0 --site yelp
python ../utils/training.py --task "rv_class" --result-dir "yelp_GALC" --model $MODEL_GALC --gpu 0 --ftype "GALC" --training  200 --norm -1 --index 1 --site yelp
python ../utils/training.py --task "rv_class" --result-dir "yelp_GALC" --model $MODEL_GALC --gpu 0 --ftype "GALC" --training  200 --norm -1 --index 2 --site yelp


python ../utils/training.py --task "rv_class" --result-dir "trip_TOPICS_64" --model $MODEL_TOPICS --gpu 0 --ftype "TOPICS_64" --training  200 --norm -1 --index 0 --site tripadvisor
python ../utils/training.py --task "rv_class" --result-dir "trip_TOPICS_64" --model $MODEL_TOPICS --gpu 0 --ftype "TOPICS_64" --training  200 --norm -1 --index 1 --site tripadvisor
python ../utils/training.py --task "rv_class" --result-dir "trip_TOPICS_64" --model $MODEL_TOPICS --gpu 0 --ftype "TOPICS_64" --training  200 --norm -1 --index 2 --site tripadvisor

python ../utils/training.py --task "rv_class" --result-dir "trip_STR" --model $MODEL_STR --gpu 0 --ftype "STR" --training  200 --norm -1 --index 0 --site tripadvisor
python ../utils/training.py --task "rv_class" --result-dir "trip_STR" --model $MODEL_STR --gpu 0 --ftype "STR" --training  200 --norm -1 --index 1 --site tripadvisor
python ../utils/training.py --task "rv_class" --result-dir "trip_STR" --model $MODEL_STR --gpu 0 --ftype "STR" --training  200 --norm -1 --index 2 --site tripadvisor

python ../utils/training.py --task "rv_class" --result-dir "trip_tfidf" --model $MODEL_TFIDF --gpu 0 --ftype "tfidf" --training  200 --norm -1 --index 0 --site tripadvisor
python ../utils/training.py --task "rv_class" --result-dir "trip_tfidf" --model $MODEL_TFIDF --gpu 0 --ftype "tfidf" --training  200 --norm -1 --index 1 --site tripadvisor
python ../utils/training.py --task "rv_class" --result-dir "trip_tfidf" --model $MODEL_TFIDF --gpu 0 --ftype "tfidf" --training  200 --norm -1 --index 2 --site tripadvisor

python ../utils/training.py --task "rv_class" --result-dir "trip_LIWC" --model $MODEL_LIWC --gpu 0 --ftype "LIWC" --training  200 --norm -1 --index 0 --site tripadvisor
python ../utils/training.py --task "rv_class" --result-dir "trip_LIWC" --model $MODEL_LIWC --gpu 0 --ftype "LIWC" --training  200 --norm -1 --index 1 --site tripadvisor
python ../utils/training.py --task "rv_class" --result-dir "trip_LIWC" --model $MODEL_LIWC --gpu 0 --ftype "LIWC" --training  200 --norm -1 --index 2 --site tripadvisor

python ../utils/training.py --task "rv_class" --result-dir "trip_INQUIRER" --model $MODEL_INQUIRER --gpu 0 --ftype "INQUIRER" --training  200 --norm -1 --index 0 --site tripadvisor
python ../utils/training.py --task "rv_class" --result-dir "trip_INQUIRER" --model $MODEL_INQUIRER --gpu 0 --ftype "INQUIRER" --training  200 --norm -1 --index 1 --site tripadvisor
python ../utils/training.py --task "rv_class" --result-dir "trip_INQUIRER" --model $MODEL_INQUIRER --gpu 0 --ftype "INQUIRER" --training  200 --norm -1 --index 2 --site tripadvisor

python ../utils/training.py --task "rv_class" --result-dir "trip_GALC" --model $MODEL_GALC --gpu 0 --ftype "GALC" --training  200 --norm -1 --index 0 --site tripadvisor
python ../utils/training.py --task "rv_class" --result-dir "trip_GALC" --model $MODEL_GALC --gpu 0 --ftype "GALC" --training  200 --norm -1 --index 1 --site tripadvisor
python ../utils/training.py --task "rv_class" --result-dir "trip_GALC" --model $MODEL_GALC --gpu 0 --ftype "GALC" --training  200 --norm -1 --index 2 --site tripadvisor

#python ../utils/training.py --task "rv_check" --result-dir "old_divider_yelp_right_norm" --model $MODEL_FC --gpu 0 --training 2000 --norm 1 --index 3 --site yelp
#python ../utils/training.py --task "rv_check" --result-dir "old_divider_yelp_right_norm" --model $MODEL_FC --gpu 0 --training 2000 --norm 1 --index 5 --site yelp

#python ../utils/training.py --task "rv_check" --result-dir "old_divider_yelp_no_norm" --model $MODEL_FC --gpu 0 --training  500 --norm -1 --index 0 --site yelp

#python ../utils/training.py --task "rv_check" --result-dir "old_divider_yelp_wrong_norm" --model $MODEL_FC --gpu 0 --training  500 --norm 1 --index 2 --site yelp
#python ../utils/training.py --task "rv_check" --result-dir "old_divider_yelp_wrong_norm" --model $MODEL_FC --gpu 0 --training  500 --norm 1 --index 4 --site yelp
#python ../utils/training.py --task "rv_check" --result-dir "old_divider_yelp_wrong_norm" --model $MODEL_FC --gpu 0 --training  500 --norm 1 --index 6 --site yelp
#python ../utils/training.py --task "rv_check" --result-dir "old_divider_yelp_wrong_norm" --model $MODEL_FC --gpu 0 --training  500 --norm 1 --index 8 --site yelp
