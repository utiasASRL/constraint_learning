#!/bin/bash
python -u _scripts/generate_all_results.py 2>&1 | tee log/generate_all_results.log
