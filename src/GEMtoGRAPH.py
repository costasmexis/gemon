import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cobra

GEM_NAME = 'textbook'

model = cobra.io.load_model(GEM_NAME)
S = cobra.util.array.create_stoichiometric_matrix(model, array_type='DataFrame')
n = S.shape[0]
m = S.shape[1]

def RAG(S):

    S = pd.DataFrame(S)
    S = S != 0
    
    for c in S.columns: S[c] = S[c].replace({True: 1, False: 0})
    
    A = S.T.dot(S)
    A = A.fillna(0)

    G = nx.from_pandas_adjacency(A)
    print("# nodes:", G.number_of_nodes(), "\n# edges:", G.number_of_edges())
    
    return A, G

def calculate_S2m(S):

    # create a new dataframe with negated values
    S_neg = S.apply(lambda x: -x)
    S_neg = S_neg.add_prefix('rev_')
    # concatenate the original dataframe and the negated dataframe
    temp_1 = pd.concat([S, S_neg], axis=1)   # [S | -S]

    # create m x m identity matrix dataframe
    identity_matrix = pd.DataFrame(np.identity(m))
    # create m x m zero matrix dataframe
    zero_matrix = pd.DataFrame(np.zeros((m,m)))
    # concatenate the two dataframes along the columns
    temp_2a = pd.concat([identity_matrix, zero_matrix], axis=1)   # [Im | 0] 
    
    reversibility = pd.DataFrame(index=S.columns)
    r = [int(model.reactions.get_by_id(rxn).reversibility) for rxn in S.columns]
    reversibility['reversibility'] = r

    diag_r = pd.DataFrame(np.diag(reversibility['reversibility']))
    temp_2b = pd.concat([zero_matrix, diag_r], axis=1)  # [0 | diag(r)]

    temp_2 = pd.concat([temp_2a, temp_2b], axis=0)
    temp_2.columns = temp_1.columns 
    temp_2.index = temp_1.columns

    S_2m = temp_1.dot(temp_2)

    return S_2m

def NFG(S):

    S_2m = calculate_S2m(S)

    S_2m_p = 1/2 * (np.abs(S_2m) + S_2m)
    S_2m_n = 1/2 * (np.abs(S_2m) - S_2m)

    ones_2m = np.ones(2 * m)

    W_p = np.linalg.pinv(np.diag(S_2m_p.dot(ones_2m)))
    W_n = np.linalg.pinv(np.diag(S_2m_n.dot(ones_2m)))

    temp_1 = (W_p.dot(S_2m_p)).T
    temp_2 = (W_n.dot(S_2m_n))

    D = pd.DataFrame(temp_1.dot(temp_2) / n)
    D.columns = S_2m.columns
    D.index = S_2m.columns
    
    G = nx.from_pandas_adjacency(D, create_using=nx.DiGraph)
    
    print("# nodes:", G.number_of_nodes(), "\n# edges:", G.number_of_edges())
    
    return D, G


def MFG(S):

    solution = model.optimize()
    v_fba = solution.fluxes.values

    v_2m_p = pd.DataFrame(1/2 * (np.abs(v_fba) + v_fba))
    v_2m_n = pd.DataFrame(1/2 * (np.abs(v_fba) - v_fba))

    v_2m = pd.concat([v_2m_p, v_2m_n])

    S_2m = calculate_S2m(S)

    S_2m_p = 1/2 * (np.abs(S_2m) + S_2m)
    S_2m_n = 1/2 * (np.abs(S_2m) - S_2m)

    j = S_2m_p.dot(v_2m)
    j = j.reshape(j.shape[0],)

    V = np.diag(v_2m[0])
    J = np.diag(j)

    t_1 = (S_2m_p.dot(V)).T

    t_2 = np.linalg.pinv(J)

    t_3 = S_2m_n.dot(V)

    t = t_1.dot(t_2)

    M = t.dot(t_3)
    M = pd.DataFame(M)

    G = nx.from_pandas_adjacency(G, create_using=nx.DiGraph)

    print("# nodes:", G.number_of_nodes(), "\n# edges:", G.number_of_edges())
    
    return M, G


def MFG(S):
        
    solution = model.optimize()
    v_fba = solution.fluxes

    v_2m_p = pd.DataFrame(1/2 * (np.abs(v_fba) + v_fba))
    v_2m_n = pd.DataFrame(1/2 * (np.abs(v_fba) - v_fba))

    S_2m = calculate_S2m(S)

    v_2m = pd.concat([v_2m_p, v_2m_n])
    v_2m.index =S_2m.columns

    S_2m_p = 1/2 * (np.abs(S_2m) + S_2m)
    S_2m_n = 1/2 * (np.abs(S_2m) - S_2m)

    j = S_2m_p.dot(v_2m)

    V = np.diag(v_2m['fluxes'])
    J = np.diag(j['fluxes'])

    t_1 = (S_2m_p.dot(V)).T

    t_2 = np.linalg.pinv(J)

    t_3 = S_2m_n.dot(V)
    t_3.columns = S_2m.columns

    t = t_1.dot(t_2)
    t.index = S_2m.columns
    t.columns = S_2m.index

    M = t.dot(t_3)
    M = pd.DataFrame(M)

    G = nx.from_pandas_adjacency(M, create_using=nx.DiGraph)
    print("# nodes:", G.number_of_nodes(), "\n# edges:", G.number_of_edges())

    return M, G
