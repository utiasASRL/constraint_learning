results_server_plot:
	- mkdir _results_server_v3
	- rsync -avz -e 'ssh' fdu@192.168.42.7:/home/fdu/constraint_learning/_results_v4/* _results_server_v3/ --exclude-from='utils/exclude-server.txt' --exclude="*.pdf"
	python _scripts/generate_all_results.py --directory="_results_server_v3"

results_small:
	python _scripts/generate_all_results.py --overwrite --directory="_results_test" --n_seeds=1

results_generate:
	python _scripts/generate_all_results.py --overwrite --directory="_results_v4" --n_seeds=3

results_generate_continue:
	python _scripts/generate_all_results.py --directory="_results_v4" --n_seeds=3
