#!/bin/bash -x

dataset=$1
epsilon=$2
times=$3

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $SCRIPT_DIR

# HDPView -- make p-view
python $SCRIPT_DIR/../src/script/make_p_view_by_hdpview.py --dataset $dataset --epsilon $epsilon --times $times

# PrivTree -- make p-view
python $SCRIPT_DIR/../src/script/make_p_view_by_privtree.py --dataset $dataset --epsilon $epsilon --times $times

# Privbayes -- synthesize data
cd $SCRIPT_DIR/../competitors/privbayes && ./privBayes.bin $dataset cq $times $epsilon && cd -
python $SCRIPT_DIR/../src/script/discretize.py --dataset $dataset --epsilon 1.0 --alg privbayes

# Evaluation
python $SCRIPT_DIR/../src/script/count_query_evaluation.py --dataset $dataset --alg all --epsilon $epsilon --times $times

echo "evaluation done"