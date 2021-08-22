from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader


def fetch_dataset(config):
    # automatic data loading and splitting
    dataset = PygGraphPropPredDataset(name=config.dataset)

    if config.feature == "full":
        pass
    elif config.feature == "simple":
        print("using simple feature")
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    split_idx = dataset.get_idx_split()

    return dataset, split_idx


def get_dataloaders(dataset, split_idx, config):
    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
