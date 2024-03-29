results_server_plot:
	- mkdir _results_server_new
	- rsync -avz -e 'ssh' fdu@192.168.42.7:/home/fdu/constraint_learning/_results_new/* _results_server_new/ --exclude-from='utils/exclude-server.txt' --exclude="*.pdf"
	python _scripts/generate_all_results.py --directory="_results_server_new"

results_generate:
	python _scripts/generate_all_results.py --overwrite --directory="_results_new"

results_generate_continue:
	python _scripts/generate_all_results.py --directory="_results_new"