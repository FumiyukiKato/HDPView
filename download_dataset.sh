#!/bin/bash
mkdir -p ./data/raw ./data/preprocessed ./data/synthetic

# bitcoin
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00526/data.zip -P ./data/raw/bitcoin
unzip ./data/raw/bitcoin/data.zip -d ./data/raw/bitcoin/

# adult
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -P ./data/raw/adult/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -P ./data/raw/adult/
sed -i -e '1d' ./data/raw/adult/adult.test

# electricity
wget https://www.openml.org/data/get_csv/2419/electricity-normalized.csv -P ./data/raw/electricity/

# phoneme
wget https://www.openml.org/data/get_csv/1592281/php8Mz7BG.csv -P ./data/raw/phoneme/

# trafic
wget http://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz -P ./data/raw/trafic
gzip -d ./data/raw/trafic/Metro_Interstate_Traffic_Volume.csv.gz 

# jm1
wget https://www.openml.org/data/get_csv/53936/jm1.csv -P ./data/raw/jm

# gowalla
wget http://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz -P ./data/raw/gowalla
gzip -d ./data/raw/gowalla/loc-gowalla_totalCheckins.txt.gz

