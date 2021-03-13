# Neural Network

## Setup

```sh
conda create --name hm-neural-network python=3.8
conda activate hm-neural-network

sh install-torch-geometric.sh
pip install -r requirements.txt

python main.py --dataset ogbg-molhiv --gnn gcn
```

## Clean

```sh
conda deactivate
conda env remove --name hm-neural-network
```
