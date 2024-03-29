#!/usr/bin/bash

# arxiv mode, verbose, compress.
mkdir _results_server_new
rsync -avz -e 'ssh' fdu@192.168.42.7:/home/fdu/constraint_learning/_results_new/* _results_server_new/ --exclude-from='_scripts/exclude-server.txt' --exclude="*.pdf"

#rsync -avz -e 'ssh' asrl@100.64.83.242:/home/asrl/research/constraint_learning/_results/* _results_laptop/ --exclude-from='_scripts/exclude-server.txt' --exclude="*.pdf"

# not needed anymore: copy starrynight dataset over to server
# rsync -avz -e 'ssh' ./starrynight fdu@192.168.42.7:/home/fdu/constraint_learning/
