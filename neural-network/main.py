import numpy as np
import torch
import torch.optim as optim
import wandb
from ogb.graphproppred import Evaluator
from tqdm import tqdm

from model.data_loader import fetch_dataset, get_dataloaders
from model.gnn import GNN
from args import get_args

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, task_type):
    model.train()
    total_loss = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            # ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(
                    pred.to(torch.float32)[is_labeled],
                    batch.y.to(torch.float32)[is_labeled],
                )
            else:
                loss = reg_criterion(
                    pred.to(torch.float32)[is_labeled],
                    batch.y.to(torch.float32)[is_labeled],
                )
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

    return total_loss


def evaluate(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    args = get_args()

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    dataset, split_idx = fetch_dataset(args)

    # automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    dataloaders = get_dataloaders(dataset, split_idx, args)
    train_loader = dataloaders["train"]
    valid_loader = dataloaders["valid"]
    test_loader = dataloaders["test"]

    if args.gnn == "gin":
        model = GNN(
            gnn_type="gin",
            num_tasks=dataset.num_tasks,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            virtual_node=False,
        ).to(device)
    elif args.gnn == "gin-virtual":
        model = GNN(
            gnn_type="gin",
            num_tasks=dataset.num_tasks,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            virtual_node=True,
        ).to(device)
    elif args.gnn == "gcn":
        model = GNN(
            gnn_type="gcn",
            num_tasks=dataset.num_tasks,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            virtual_node=False,
        ).to(device)
    elif args.gnn == "gcn-virtual":
        model = GNN(
            gnn_type="gcn",
            num_tasks=dataset.num_tasks,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            virtual_node=True,
        ).to(device)
    else:
        raise ValueError("Invalid GNN type")

    with wandb.init(project="hongbomiao.com", entity="hongbo-miao", config=args) as wb:
        wb.watch(model)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        valid_curve = []
        test_curve = []
        train_curve = []

        for epoch in range(1, args.epochs + 1):
            print(f"=====Epoch {epoch}")
            print("Training...")
            train_loss = train(
                model, device, train_loader, optimizer, dataset.task_type
            )

            print("Evaluating...")
            train_perf = evaluate(model, device, train_loader, evaluator)
            valid_perf = evaluate(model, device, valid_loader, evaluator)
            test_perf = evaluate(model, device, test_loader, evaluator)

            print({"Train": train_perf, "Validation": valid_perf, "Test": test_perf})
            wb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_perf,
                    "val_acc": valid_perf,
                    "test_acc": test_perf,
                }
            )

            train_curve.append(train_perf[dataset.eval_metric])
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])

        if "classification" in dataset.task_type:
            best_val_epoch = np.argmax(np.array(valid_curve))
            best_train = max(train_curve)
        else:
            best_val_epoch = np.argmin(np.array(valid_curve))
            best_train = min(train_curve)

        print("Finished training!")
        print(f"Best validation score: {valid_curve[best_val_epoch]}")
        print(f"Test score: {test_curve[best_val_epoch]}")

        if not args.filename == "":
            torch.save(
                {
                    "Val": valid_curve[best_val_epoch],
                    "Test": test_curve[best_val_epoch],
                    "Train": train_curve[best_val_epoch],
                    "BestTrain": best_train,
                },
                args.filename,
            )


if __name__ == "__main__":
    main()
