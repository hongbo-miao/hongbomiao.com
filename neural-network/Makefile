setup:
	sh bin/setup.sh

clean:
	sh bin/clean.sh

train:
	python main.py --dataset ogbg-molhiv --gnn gcn

debug:
	python main.py --dataset ogbg-molhiv --gnn gcn --feature simple --epochs 30
