import numpy as np
import hartree_fock


f_name = 'H2O__2Rref_Helgaker_6-31G.npz'
integrais = np.load(f_name)
h = integrais['h']
g = integrais['g']
S = integrais['S']
Vnuc = integrais['Vnuc']

N = 5

eps = 1e-5
nmax = 30

E, C = hartree_fock.hf_restrito(h, g, S, N, eps, nmax)

print(f'Energia total: {E + Vnuc:.8f}')

