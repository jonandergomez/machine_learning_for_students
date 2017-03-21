#!/bin/bash

export PYTHONPATH="${HOME}/python"

#python test_MLE.py --standalone

#spark-submit --master local[4] test_MLE.py

spark-submit --master spark://eip6029.inf.upv.es:7077 \
             --executor-memory 6G \
             --py-files ${PYTHONPATH}/mypythonlib.tgz \
             test_MLE.py  --num-slices 80  --dataset etc/rodrigo-index-raw5.txt
