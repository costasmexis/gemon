import pandas as pd
import numpy as np
import torch
import networkx as nx
import pickle

# LOAD
G = pickle.load(open('yeast_G.pickle', 'rb'))

# Load ORACLE data
sigma = pd.read_csv('saturations.csv', index_col=0)
gamma = pd.read_csv('gamma.csv', index_col=0)
vmax = pd.read_csv('Vmax_matrix.csv', index_col=0)

# Keep data for only one Kinetic Model
gamma = gamma.head(1)
sigma = sigma.head(1)
vmax = vmax.head(1)

gamma = gamma.T
gamma.rename(columns = {'2,46':'value'}, inplace=True)

# # reaction to remove from graph because gamma > 1
# rxn_to_remove = gamma[gamma['value'] > 1].index.values
# G.remove_nodes_from(list(rxn_to_remove))
# print("# nodes:", G.number_of_nodes(), "\n# edges:", G.number_of_edges())


# # get the reactions that are the reversible 
# rev_rxn = []
# for node in list(G.nodes()):
#     if node.split("?")[0] == 'rev': rev_rxn.append(node.split("?")[1])

# # rename the reactions of gamma; if it's the reversible one add rev? to the column name
# for col in gamma.columns:
#     if col in rev_rxn: gamma.rename(columns={col:'rev?'+col}, inplace=True)


# for node in gamma.columns:
#     try:
#         G.nodes[node]['gamma'] =  gamma[node].values[0]
#     except KeyError:
#         pass

# no_gamma_nodes = [node for node, data in G.nodes(data=True) if not data]

# for node in no_gamma_nodes: G.nodes[node]['gamma'] = np.nan


# df = pd.DataFrame(index=list(G.nodes()), columns=['compartements', 'metabolites', 'num_of_mets'])

# for rxn in df.index:
#     if 'rev?' in rxn: rxn = rxn.split("?")[1]
#     else: rxn = rxn

#     metabolites = []
#     for m in model.reactions.get_by_id(rxn).metabolites: metabolites.append(m.id)
  
#     metabolites = "|".join(metabolites)
#     compartements = "|".join(list(model.reactions.get_by_id(rxn).compartments))

#     # compartements = list(model.reactions.get_by_id(rxn).compartments)
    
#     num_of_mets = len(model.reactions.get_by_id(rxn).metabolites)
#     new_row = [compartements, metabolites, num_of_mets]

#     df.loc[rxn] = new_row

# # rename 'gamma' feature name to 'y'
# for node, data in G.nodes(data=True):
#     data['y'] = data.pop('gamma')

# for node in list(G.nodes()):
#     if 'rev?' in node: rxn = node.split("?")[1]
#     else: rxn = node

#     # nx.set_node_attributes(G, {nodata.xde: {'compartements':df.loc[rxn]['compartements']}})
#     nx.set_node_attributes(G, {node: {'num_of_mets':df.loc[rxn]['num_of_mets']}})

# import torch
# import torch.nn as nn
# from torch_geometric.utils.convert import from_networkx

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data = from_networkx(G, group_edge_attrs=all)

# print(data)
# print()
# print(data.num_nodes ,data.num_edges)

# data.x = data.num_of_mets
# del data.num_of_mets

# ###### Split to train and test ######
# N = data.x.shape[0]

# train_idx = int(N * .8)

# train_mask = torch.zeros(N, dtype=torch.bool)
# test_mask = torch.zeros(N, dtype=torch.bool)

# train_mask[:train_idx] = 1
# test_mask[train_idx:] = 1

# data.train_mask = train_mask
# data.test_mask = test_mask
