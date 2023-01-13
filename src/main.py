#%%
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx
from networkx import from_pandas_edgelist
%matplotlib inline

SEED = 42

df = pd.read_csv('../data/queries/query-result-full.csv')
print(df.head())
print(df.shape)

G = from_pandas_edgelist(df, source='s', target='o', 
                            create_using=nx.DiGraph)
print('Nodes:', G.number_of_nodes())
print('Edges:', G.number_of_edges())
# %%
''' Directed / Undirected Graph'''

if nx.is_directed(G): print('Graph IS DIRECTED.')
else: print('Graph IS UNDIRECTED')
# %%
def plot_graph():Applied Data Science with Python Specialization
    plt.figure(figsize=(50, 25), dpi=80)
    nx.draw_networkx(G, with_labels=False, node_size=100)
    plt.show()
# %%
print("The Graph's density is", round(nx.density(G), 4))
# %%
pageRank = pd.DataFrame.from_dict(nx.pagerank(G), orient='index').sort_values(by=0, ascending=False)
pageRank = pageRank.rename(columns={0:'PageRank'})

print('Top 10 nodes by PageRank:')
display(pageRank.head(10))

print('Last 10 nodes by PageRank:')
display(pageRank.tail(10))
# %%
''' Betweenness Centrality '''
betCen = nx.betweenness_centrality(G)
betCen = pd.DataFrame.from_dict(nx.betweenness_centrality(G, seed=SEED), orient='index').sort_values(by=0, ascending=False)
betCen = betCen.rename(columns={0:'BC'})

print('Top 10 nodes by Betweenness Centrality:')
display(betCen.head(10))

print('Last 20 nodes by Betweenness Centrality:')
display(betCen.tail(20))
# %%
''' Eigenvector Centrality '''
eigCen = pd.DataFrame.from_dict(nx.eigenvector_centrality(G), orient='index').sort_values(by=0, ascending=False)
eigCen = eigCen.rename(columns={0:'EC'})

print('Top 10 nodes by Eigenvector Centrality:')
display(eigCen.head(10))

print('Last 20 nodes by Eigenvector Centrality:')
display(eigCen.tail(20))
# %%
