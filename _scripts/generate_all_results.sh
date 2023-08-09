#/usr/bin/bash

conda env create -f environment.yml
conda activate lifting

python _scripts/run_range_only_study.pdf
python _scripts/run_stereo_study.pdf
python _scripts/run_other_study.pdf
python _scripts/run_all_study.pdf
