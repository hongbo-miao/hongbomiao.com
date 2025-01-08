import torch
from model.conv import GNNNode, GNNVirtualNode
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import (
    GlobalAttention,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


class GNN(torch.nn.Module):
    def __init__(
        self,
        num_tasks: int,
        num_layer: int = 5,
        emb_dim: int = 300,
        gnn_type: str = "gin",
        virtual_node: bool = True,
        residual: bool = False,
        drop_ratio: float = 0.5,
        jk: str = "last",
        graph_pooling: str = "mean",
    ) -> None:
        super().__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.jk = jk
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            msg = "Number of GNN layers must be greater than 1."
            raise ValueError(msg)

        # GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNNVirtualNode(
                num_layer,
                emb_dim,
                jk=jk,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )
        else:
            self.gnn_node = GNNNode(
                num_layer,
                emb_dim,
                jk=jk,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    torch.nn.BatchNorm1d(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * emb_dim, 1),
                ),
            )
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            msg = "Invalid graph pooling type."
            raise ValueError(msg)

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data: Data) -> Tensor:
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        return self.graph_pred_linear(h_graph)
