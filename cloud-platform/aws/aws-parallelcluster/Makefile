poetry-env-use:
	poetry env use python3.12
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

poetry-run-test:
	poetry run poe test
poetry-run-test-coverage:
	poetry run poe test-coverage

pcluster-create:
	pcluster create-cluster --cluster-name=hm-hpc-cluster --cluster-configuration=config/hm-hpc-cluster-config.yaml
pcluster-update:
	pcluster update-cluster --cluster-name=hm-hpc-cluster --cluster-configuration=config/hm-hpc-cluster-config.yaml --dryrun=true
	# Stop
	pcluster update-compute-fleet --cluster-name=hm-hpc-cluster --status=STOP_REQUESTED
	pcluster describe-compute-fleet --cluster-name=hm-hpc-cluster
	# Update
	pcluster update-cluster --cluster-name=hm-hpc-cluster --cluster-configuration=config/hm-hpc-cluster-config.yaml
	pcluster describe-cluster --cluster-name=hm-hpc-cluster
	# Start
	pcluster update-compute-fleet --cluster-name=hm-hpc-cluster --status=START_REQUESTED
	pcluster describe-compute-fleet --cluster-name=hm-hpc-cluster
pcluster-delete:
	pcluster delete-cluster --cluster-name=hm-hpc-cluster
	aws cloudformation list-stacks
	aws cloudformation delete-stack --stack-name=parallelclusternetworking-pubpriv-xxxxxxxxxxxxxx
pcluster-list:
	pcluster list-clusters
pcluster-describe:
	pcluster describe-cluster --cluster-name=hm-hpc-cluster
pcluster-ssh:
	pcluster ssh --cluster-name=hm-hpc-cluster
pcluster-dcv-connect:
	pcluster dcv-connect --cluster-name=hm-hpc-cluster

slurm-job-submit:
	sbatch --wrap="srun jobs/hello.sh"
	sbatch --wrap="srun jobs/hello.sh" --nodes=2
	sbatch --wrap="srun jobs/hello.sh" --nodes=3 --partition=spot-queue --constraint="[c7gn.16xlarge*1&c7gn.metal*2]"
slurm-queue-view:
	squeue
slurm-module-list-available:
	module avail
slurm-module-list-loaded:
	module list

open-mpi-version:
	mpirun --version
