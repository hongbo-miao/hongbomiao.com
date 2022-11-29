conda-create:
	conda create --name=hm-airflow python=3.10 --yes
conda-env-remove:
	conda env remove --name=hm-airflow
conda-activate:
	conda activate hm-airflow
conda-deactivate:
	conda deactivate
conda-env-list:
	conda env list
conda-list-packages:
	conda list

pip-install:
	pip install "apache-airflow[celery]==2.3.3" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.3.3/constraints-3.10.txt"

helm-download-values:
	helm show values apache-airflow/airflow > values.yaml
helm-get-my-values:
	helm get values airflow --namespace=hm-airflow
