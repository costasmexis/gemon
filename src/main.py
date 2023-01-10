from cobrapy_bigg_client import client

bigg_model = client.download_model("e_coli_core", save=False)
print(bigg_model)  # => cobra.core.model.Model

import networkx as nx

G = nx.DiGraph()

for node in bigg_model.metabolites:
    G.add_node(node.id, name=node.name)


# Add the edges (reactions) to the graph
for edge in bigg_model.reactions:
    for metabolite in edge.metabolites:
        G.add_edge(metabolite, edge.id)

from torch_geometric.data import Data

edge_index = torch.tensor(list(G.edges()), dtype=torch.long)
x = torch.tensor([G.nodes[i]["name"] for i in G.nodes], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

import torch.nn as nn

# choose any GNN model from torch_geometric.nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, data.num_node_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.x)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    node_representations = model(data).detach()

# Use some clustering algorithm like Kmeans to group similar nodes together
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(node_representations)

# Create a new graph with a reduced number of nodes
compressed_graph = nx.DiGraph()
for i, cluster in enumerate(clusters):
    compressed_graph.add_node(cluster, name=G.nodes[i]["name"])

# Add edges to the compressed graph
for i, cluster1 in enumerate(clusters):
    for j, cluster2 in enumerate(clusters):
        if G.has_edge(i, j):
            compressed