import networkx as nx
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader

# Load Zachary's Karate Club graph
g = nx.karate_club_graph()

# Prepare the node features and labels
x = torch.tensor([[1 if i in g.nodes[j]['club'] else 0 for i in range(2)] for j in g.nodes], dtype=torch.float)
y = torch.tensor([g.nodes[i]['club'] for i in g.nodes], dtype=torch.long)

# Prepare the edge index and edge attributes
edge_index = torch.tensor([[i, j] for i, j in g.edges], dtype=torch.long)
edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float)

# Create a PyTorch Geometric Data object
data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

# Define the GNN model
class GNN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GNN, self).__init__()
        self.conv1 = nn.Linear(in_feats, hidden_size)
        self.conv2 = nn.Linear(hidden_size, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)

# Initialize the model
gnn = GNN(in_feats=2, hidden_size=5, num_classes=2)

# Define the loss function and optimizer
loss_func = nn.NLLLoss()
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)

# Training the model
for epoch in range(100):
    optimizer.zero_grad()
    out = gnn(data)
    loss = loss_func(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
