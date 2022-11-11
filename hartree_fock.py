import numpy as np
from scipy.linalg import eigh

def imprimir_energia(E, i):
    """Imprime a energia e a iteração
    
    Parâmetros:
    E - energia (float)
    i - iteração (int)

    Retorna:
    Não retorna nada
    
    """
    print(f'{i}\t {E:.8f}')


def densidade_inicial(n):
    """Calcula um chute inicial para a densidade

    Parâmetros:
    n - tamanho da base
    
    Retorna
    A densidade inicial
    """
    return np.zeros((n, n))


def calcular_densidade(C):
    """Calcula a densidade a partir dos coeficietes

    Parâmetros:
    Coeficientes dos orbitais (2D array)
    
    Retorna:
    A matriz densidade 
    
    """
    return 2 * C @ C.T


def convergiu(P, Pold, eps):
    """Verifica se P e Pold são próximas
    
    Parâmetros:
    P    - A matriz de densidade
    Pold - A matriz de densidade antiga
    eps  - Critério
    
    Retorna:
    True ou False, dependendo se P e Pold são próximas
    
    """
    return np.allclose(P, Pold, atol=eps)


def calcular_energia(P, h, F):
    """Calcula a energia a partir das matrizes na base
    
    Parâmetros:
    P - matriz de densidade
    h - matriz com as integrais de um elétron
    F - matriz de Fock
    
    Retorna:
    A energia (float)
    
    """
    n = len(h)
    E = 0.0
    for i in range(n):
        for j in range(n):
            E += P[i, j] * (h[i, j] + F[i, j])
    return E/2
## (np.einsum('ij,ij', P, h) + np.einsum('ij,ij', P, F))/2


def calcular_fock(P, g, h):
    """Calcula a matriz de Fock
    
    Parâmetros:
    P - matriz de densidade
    g - matriz com as integrais de dois elétrons
    h - matriz com as integrais de um elétron
    
    Retorna:
    A matriz de Fock (2D array)
    
    """
    n = len(h)
    v = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    v[i,j] += P[k, l] *(g[i,j,k,l]
                                        - g[i,l,k,j]/2)
    return h + v


def diagonalizar_fock(F, S, N):
    """Diagonaliza a matriz de Fock
    
    Parâmetros:
    F - matriz de Fock
    S - matriz com as integrais de overlap
    N - Número de orbitais ocupados
    
    Retorna:
    Os N primeiros autovetores de F
    
    """
    e_orb, C = eigh(F, S)
    return C[:, :N]



def hf_restrito(h, g, S, N, eps, nmax):
    """Calcula hartree fock restrito

    Parâmetros:
    h  - integrais de um elétron (2D array)
    g  - integrais de dois elétrons (2D array)
    S  - integrais de um sobreposição (2D array)
    N  - Número de orbitais ocupados (int)
    eps - critério de convergência (float)
    nmax - número máximo de iterações (int)
    
    Retorna
    a energia (float) e os coeficientes dos orbitais (2D array)
    """
    n = np.shape(h)[0]
    P = densidade_inicial(n)
    for i in range(nmax):
        F = calcular_fock(P, g, h)
        E = calcular_energia(P, h, F)
        imprimir_energia(E, i)
        C = diagonalizar_fock(F, S, N)
        Pold = P
        P = calcular_densidade(C)
        if convergiu(P, Pold, eps):
            return E, C
