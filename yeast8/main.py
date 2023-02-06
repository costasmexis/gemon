import pandas as pd
import numpy as np
import torch
import networkx as nx
import cobra
import matplotlib.pyplot as plt

import sys
sys.path.append("../src/")
import GEMtoGRAPH as gg

pd.set_option('display.float_format', lambda x: '%.5f' % x)

model = cobra.io.load_json_model('redYeast_ST8943_fdp1.json')
S = cobra.util.array.create_stoichiometric_matrix(model, array_type='DataFrame')

# load tfa fluxes and send them to graph construction functions
tfa = pd.read_csv('fluxes_for_graph.csv', index_col=0)
tfa = tfa.head(1)

zero_flux = [col for col in tfa.columns if (tfa[col] == 0).all()]

print('Zero flux reactions:',len(zero_flux))

tfa.drop(columns=zero_flux, inplace=True)
print("TFA fluxes:", tfa.shape[1])

# For _reverse reactions we should change the sign of the flux to negative
for col in tfa.columns:
    if '_reverse' in col: tfa[col] = -tfa[col]


tfa.rename(columns={col: col.split("_reverse_")[0] for col in tfa.columns}, inplace=True)

tfa_flux = tfa.iloc[0].values
tfa_flux = pd.DataFrame(columns=['fluxes'], data=tfa_flux)
tfa_flux.index = S.columns

M, S_2m, G = gg.MFG(S, model, tfa_flux)

# Remove isolated nodes from G
print()
print('Removing isolated nodes...')
isolated_nodes = list(nx.isolates(G))
G.remove_nodes_from(isolated_nodes)

print("# nodes:", G.number_of_nodes(), "\n# edges:", G.number_of_edges())

sigma = pd.read_csv('saturations.csv', index_col=0)
gamma = pd.read_csv('gamma.csv', index_col=0)
vmax = pd.read_csv('Vmax_matrix.csv', index_col=0)

gamma = gamma.head(1)
sigma = sigma.head(1)
vmax = vmax.head(1)

# get the reactions that are the reversible 
rev_rxn = []
for node in list(G.nodes()):
    if node.split("?")[0] == 'rev': rev_rxn.append(node.split("?")[1])

# rename the reactions of gamma; if it's the reversible one add rev? to the column name
for col in gamma.columns:
    if col in rev_rxn: gamma.rename(columns={col:'rev?'+col}, inplace=True)


for node in gamma.columns:
    try:
        G.nodes[node]['gamma'] =  gamma[node].values[0]
    except KeyError:
        pass

no_gamma_nodes = [node for node, data in G.nodes(data=True) if not data]

for node in no_gamma_nodes: G.nodes[node]['gamma'] = np.nan

# drop these lines because of gamma > 1
index_to_drop = df_g[df_g['gamma'] > 1].index

df_g.drop(index=index_to_drop, inplace=True)

