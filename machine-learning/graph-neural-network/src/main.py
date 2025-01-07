import numpy as np
import torch
import wandb
from args import get_args
from model.data_loader import fetch_dataset, get_dataloaders
from model.gnn import GNN
from ogb.graphproppred import Evaluator
from torch import optim
from tqdm import tqdm

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, task_type):
    model.train()
    total_loss = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        device_batch = batch.to(device)

        if device_batch.x.shape[0] == 1 or device_batch.batch[-1] == 0:
            pass
        else:
            pred = model(device_batch)
            optimizer.zero_grad()
            # ignore nan targets (unlabeled) when computing training loss.
            is_labeled = device_batch.y == device_batch.y
            if "classification" in task_type:
                loss = cls_criterion(
                    pred.to(torch.float32)[is_labeled],
                    device_batch.y.to(torch.float32)[is_labeled],
                )
            else:
                loss = reg_criterion(
                    pred.to(torch.float32)[is_labeled],
                    device_batch.y.to(torch.float32)[is_labeled],
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
        device_batch = batch.to(device)

        if device_batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(device_batch)

            y_true.append(device_batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    args = get_args()

    with wandb.init(
        entity="hongbo-miao",
        project="graph-neural-network",
        config=args,
    ) as wb:
        config = wb.config

        device = (
            torch.device("cuda:" + str(config.device))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        dataset, split_idx = fetch_dataset(config)

        # automatic evaluator. takes dataset name as input
        evaluator = Evaluator(config.dataset)

        dataloaders = get_dataloaders(dataset, split_idx, config)
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        test_loader = dataloaders["test"]

        match config.gnn:
            case "gin":
                model = GNN(
                    gnn_type="gin",
                    num_tasks=dataset.num_tasks,
                    num_layer=config.num_layer,
                    emb_dim=config.emb_dim,
                    drop_ratio=config.drop_ratio,
                    virtual_node=False,
                ).to(device)
            case "gin-virtual":
                model = GNN(
                    gnn_type="gin",
                    num_tasks=dataset.num_tasks,
                    num_layer=config.num_layer,
                    emb_dim=config.emb_dim,
                    drop_ratio=config.drop_ratio,
                    virtual_node=True,
                ).to(device)
            case "gcn":
                model = GNN(
                    gnn_type="gcn",
                    num_tasks=dataset.num_tasks,
                    num_layer=config.num_layer,
                    emb_dim=config.emb_dim,
                    drop_ratio=config.drop_ratio,
                    virtual_node=False,
                ).to(device)
            case "gcn-virtual":
                model = GNN(
                    gnn_type="gcn",
                    num_tasks=dataset.num_tasks,
                    num_layer=config.num_layer,
                    emb_dim=config.emb_dim,
                    drop_ratio=config.drop_ratio,
                    virtual_node=True,
                ).to(device)
            case _:
                raise ValueError("Invalid GNN type")

        wb.watch(model)

        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        val_curve = []
        test_curve = []
        train_curve = []
        max_perf_metric_val = 0.0

        for epoch in range(1, config.epochs + 1):
            print(f"=====Epoch {epoch}")
            print("Training...")
            train_loss = train(
                model,
                device,
                train_loader,
                optimizer,
                dataset.task_type,
            )

            print("Evaluating...")
            train_perf = evaluate(model, device, train_loader, evaluator)
            val_perf = evaluate(model, device, val_loader, evaluator)
            test_perf = evaluate(model, device, test_loader, evaluator)

            print({"Train": train_perf, "Validation": val_perf, "Test": test_perf})
            wb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_perf": train_perf,
                    "val_perf": val_perf,
                    "test_perf": test_perf,
                },
            )

            train_curve.append(train_perf[dataset.eval_metric])
            val_curve.append(val_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])

            # Save model
            if val_perf[dataset.eval_metric] > max_perf_metric_val:
                print("Found better model.")
                max_perf_metric_val = val_perf[dataset.eval_metric]
                torch.save(model.state_dict(), "model.pt")
                wb.save("model.pt")

        if "classification" in dataset.task_type:
            best_val_epoch = np.argmax(np.array(val_curve))
            best_train = max(train_curve)
        else:
            best_val_epoch = np.argmin(np.array(val_curve))
            best_train = min(train_curve)

        print("Finished training!")
        print(f"Best validation score: {val_curve[best_val_epoch]}")
        print(f"Test score: {test_curve[best_val_epoch]}")

        if not config.filename == "":
            torch.save(
                {
                    "Val": val_curve[best_val_epoch],
                    "Test": test_curve[best_val_epoch],
                    "Train": train_curve[best_val_epoch],
                    "BestTrain": best_train,
                },
                config.filename,
            )


if __name__ == "__main__":
    main()
