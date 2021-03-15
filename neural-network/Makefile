setup:
	sh bin/setup.sh

clean:
	sh bin/clean.sh

train:
	python main.py --dataset ogbg-molhiv --gnn gcn

train-deeper-gcn:
	python main.py --dataset ogbg-molhiv --gnn deeper-gcn --conv_encode_edge --num_layers 7 --dataset ogbg-molhiv --block res+ --gcn_aggr softmax --t 1.0 --learn_t --drop_ratio 0.2

debug:
	python main.py --dataset ogbg-molhiv --gnn gcn --feature simple --epochs 30
