import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import degree


# GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(GINConv, self).__init__(aggr="add")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp(
            (1 + self.eps) * x
            + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


# GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr="add")
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        # x: [N, in_channels]
        # edge_index: [2, E]
        # edge_attr: [E, 2]
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)  # [E, out_channels]

        n1, n2 = edge_index  # [E], [E]

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(n1, x.size(0), dtype=x.dtype) + 1  # [N]
        deg_inv_sqrt = deg.pow(-0.5)  # [N]
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[n1] * deg_inv_sqrt[n2]  # [N]

        return self.propagate(
            edge_index, x=x, edge_attr=edge_embedding, norm=norm
        ) + F.relu(x + self.root_emb.weight) * 1.0 / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        # x_j: [E, out_channels]
        # edge_attr: [E, out_channels]
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        # aggr_out: [N, out_channels]
        return aggr_out


# GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(
        self,
        num_layer,
        emb_dim,
        drop_ratio=0.5,
        jk="last",
        residual=False,
        gnn_type="gin",
    ):
        """
        emb_dim (int): node embedding dimensionality
        num_layer (int): number of GNN message passing layers
        """

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.jk = jk
        # add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for _ in range(num_layer):
            if gnn_type == "gin":
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == "gcn":
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError(f"Undefined GNN type called {gnn_type}")

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
        )

        # computing input node embedding
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            # h has shape [N, in_channels]
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        # Different implementations of JK-concat
        if self.jk == "last":
            node_representation = h_list[-1]
        elif self.jk == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]

        return node_representation


# Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(
        self,
        num_layer,
        emb_dim,
        drop_ratio=0.5,
        jk="last",
        residual=False,
        gnn_type="gin",
    ):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.jk = jk
        # add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        # set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        # batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        # List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == "gin":
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == "gcn":
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError(f"Undefined GNN type called {gnn_type}")

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    torch.nn.BatchNorm1d(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * emb_dim, emb_dim),
                    torch.nn.BatchNorm1d(emb_dim),
                    torch.nn.ReLU(),
                )
            )

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )

        # virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
        )

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            # add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            # Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            # update the virtual nodes
            if layer < self.num_layer - 1:
                # add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = (
                    global_add_pool(h_list[layer], batch) + virtualnode_embedding
                )
                # transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training,
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training,
                    )

        # Different implementations of JK-Concat
        if self.jk == "last":
            node_representation = h_list[-1]
        elif self.jk == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]

        return node_representation
