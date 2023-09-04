#!/usr/bin/bash

# arxiv mode, verbose, compress.
rsync -avz -e 'ssh' fdu@192.168.42.7:/home/fdu/constraint_learning/_results/* _results_server/ --exclude-from='_scripts/exclude-server.txt'

# not needed anymore: copy starrynight dataset over to server
# rsync -avz -e 'ssh' ./starrynight fdu@192.168.42.7:/home/fdu/constraint_learning/
