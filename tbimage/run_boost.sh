#!/usr/bin/env bash

if [ $# -eq 0 ]
then
    echo "specify operation type (train, predict, train_in_svm, in_svm, cv) etc"
fi

today=`date +"%Y%m%d"`
train_data="iq_train_data.csv"
train_svm="iq_train_data_svm.txt"
test_data="iq_test_data.csv"
test_predict="iq_test_predict.csv"
model_name="xgbtree_${today}.model"

o_type="$1"
if [[ $o_type = "train" ]]
then
    python gbdt_classifier.py train ${train_data} ${model_name} --nfold=8 --xindex=2 --max_depth=6 --silent=1 --num_class=4 --num_round=500 --min_child_weight=30 --eta=0.1
elif [[ $o_type = "train_in_svm" ]]
then
    python gbdt_classifier.py train_in_svm ${train_svm} svm_${model_name} --split --nfold=8 --num_round=2
elif [[ $o_type = "to_svm" ]]
then
    python gbdt_classifier.py to_svm ${train_data} ${train_svm} --sep=, --xindex=2
elif [[ $o_type = "predict" ]]
then
    nohup python gbdt_classifier.py predict ${test_data} ${test_predict} --model=${model_name} --nthread=10 --batch_size=150000 --xindex=1 &
elif [[ $o_type = "sampling" ]]
then
    python gbdt_classifier.py sampling ${test_predict} iq_test_predict_sample.csv --sampling_size=1000
elif [[ $o_type = "cv" ]]
then
    python gbdt_classifier.py cv ${train_svm} no_output_needed --nfold=8 --num_round=800
elif [[ $o_type = "grid" ]]
then
    nohup python -u gbdt_classifier.py grid ${train_svm}.train no_output_needed --test=${train_svm}.test > grid.log &
fi
