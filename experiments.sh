# !/bin/bash

# python3 script.py 1 150 10 euclidean iris
# python3 script.py 1 100 10 cosine iris
# python3 script.py 1 100 10 cityblock iris
python3 script.py 1 50 10 euclidean ds2c2sc13.arff
python3 script.py 2 50 10 euclidean ds2c2sc13.arff
python3 script.py 3 50 10 euclidean ds2c2sc13.arff
python3 script.py 4 50 10 euclidean ds2c2sc13.arff
python3 script.py 5 50 10 euclidean ds2c2sc13.arff
# python3 script.py 6 50 12 euclidean ds2c2sc13.arff
# python3 script.py 7 50 12 euclidean ds2c2sc13.arff
# python3 script.py 8 50 12 euclidean ds2c2sc13.arff
# python3 script.py 9 50 12 euclidean ds2c2sc13.arff
# python3 script.py 10 50 12 euclidean ds2c2sc13.arff
# python3 script.py 11 50 12 euclidean ds2c2sc13.arff
# python3 script.py 12 50 12 euclidean ds2c2sc13.arff
# python3 script.py 13 50 12 euclidean ds2c2sc13.arff
# python3 script.py 14 50 12 euclidean ds2c2sc13.arff
# python3 script.py 15 50 12 euclidean ds2c2sc13.arffs
# python3 script.py 1 50 10 cosine 2sp2glob.arff
# python3 script.py 1 50 10 cityblock 2sp2glob.arff
# python3 script.py 1 50 10 euclidean complex8.arff
# python3 script.py 1 50 10 cosine complex8.arff
# python3 script.py 1 50 10 cityblock complex8.arff
# python3 script.py 1 50 10 euclidean complex9.arff
# python3 script.py 1 50 10 cosine complex9.arff
# python3 script.py 1 50 10 cityblock complex9.arff
# python script.py 1 100 30 euclidean,cosine,cityblock iris

# seeds=$(seq 1 5)
# distances="euclidean,cosine,cityblock"
# size=30
# reps=10
# datasets="iris blobs"

# for dataset in $datasets; do
#   (
#     for seed in $seeds; do
#       python3 script.py $seed $size $reps $distances $dataset
#     done
#   ) &
# done

# wait
# echo "All experiments finished."