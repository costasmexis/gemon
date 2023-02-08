import pandas as pd
import numpy as np
import torch
import networkx as nx
import cobra

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

" **** SAVE NETWORKX GRAPH TO FILE"
import pickle

# SAVE
pickle.dump(G, open('yeast_G.pickle', 'wb'))

# LOAD
# G = pickle.load(open('yeast_G.pickle', 'rb'))