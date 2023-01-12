#%%
import cobra
from cobra.io import load_model

model = load_model('textbook')
display(model)
# %%
from cobra.flux_analysis import fastcc

fcc = fastcc(model)
# %%
''' Reactions deleted using FASTCC '''
original_rxn = []
for rxn in model.reactions: original_rxn.append(rxn.id)

fastcc_rxn = []
for rxn in fcc.reactions: fastcc_rxn.append(rxn.id)

deleted_rxn = set(original_rxn) - set(fastcc_rxn)

print('# of deleted reactions:', len(deleted_rxn))
print(deleted_rxn)
# %%

