#!/bin/bash

TIMESTAMP=`date +"%Y-%m-%d-%H-%M-%S"`
echo $TIMESTAMP
python -u _scripts/generate_all_results.py 2>&1 | tee log/generate_all_results_$TIMESTAMP.log
