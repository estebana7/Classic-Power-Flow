#=================================================== TRABALHO 1 ===================================================
#Trimestre: 2023.1
#Disciplina: 210001 - Análise de Redes I
#Docente: João Alberto Passos Filho
#Discente: Esteban Vicente Aguilar Bojorge
#Enunciado: Desenvolver um programa em MatLab ou Python para determinar a matriz de admitância nodal (Y_barra) do
#           sistema de 24-barras, cujos dados elétrinp.cos estão fornecidos em anexo no formato do programa Anarede.
#==================================================================================================================

import time
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
np.set_printoptions(suppress=True)

#%%Dados adaptados do arquivo do editCEPEL 'IEEE 24 Barras.pwf'

#DBAR
#         0     1       2      3       4      5       6     7    8      9   10      11   12     13
#       (Num) Tipo     (V) (Angl)   (Pg)    (Qg)   (Qn)  (Qm) (Bc)   (Pl) (Ql)    (Sh)  Are   (Vf)
D_BAR = [[ 1,   1,   1.035, -22.,  172.,  24.81,  -50.,  80.,   0,  108., 22.,      0,  1,   1000],
         [ 2,   1,   1.035, -22.,  172.,   17.1,  -50.,  80.,   0,   97., 20.,      0,  1,   1000],
         [ 3,   3,   1.000, -20.,     0,      0,     0,    0,   0,  180., 37.,      0,  1,   1000],
         [ 4,   3,   0.998, -24.,     0,      0,     0,    0,   0,   74., 15.,      0,  1,   1000],
         [ 5,   3,   1.017, -24.,     0,      0,     0,    0,   0,   71., 14.,      0,  1,   1000],
         [ 6,   3,   1.010, -27.,     0,      0,     0,    0,   0,  136., 28.,  -100.,  1,   1000],
         [ 7,   1,   1.025, -21.,  240.,  53.09,    0., 180.,   0,  125., 25.,      0,  1,   1000],
         [ 8,   3,   0.992, -25.,     0,      0,     0,    0,   0,  171., 35.,      0,  1,   1000],
         [ 9,   3,   1.000, -22.,     0,      0,     0,    0,   0,  175., 36.,      0,  1,   1000],
         [10,   3,   1.000, -24.,     0,      0,     0,    0,   0,  195., 40.,      0,  1,   1000],
         [11,   3,   0.990, -16.,     0,      0,     0,    0,   0,     0,   0,      0,  1,   1000],
         [12,   3,   1.003, -15.,     0,      0,     0,    0,   0,     0,   0,      0,  1,   1000],
         [13,   2,   1.020, -13., 285.3,  117.5,    0., 240.,   0,  265., 54.,      0,  2,   1000],
         [14,   1,   0.980, -13.,    0.,  -36.5,  -50., 200.,   0,  194., 39.,      0,  2,   1000],
         [15,   1,   1.014, -4.8,  215.,  -23.6,  -50., 110.,   0,  317., 64.,      0,  2,   1000],
         [16,   1,   1.017, -5.5,  155.,  32.79,  -50.,  80.,   0,  100., 20.,      0,  2,   1000],
         [17,   3,   1.039, -1.6,     0,      0,     0,    0,   0,     0,   0,      0,  2,   1000],
         [18,   1,   1.050, -.56,  400.,  134.4,  -50., 200.,   0,  333., 68.,      0,  2,   1000],
         [19,   3,   1.023, -6.4,     0,      0,     0,    0,   0,  181., 37.,      0,  2,   1000],
         [20,   3,   1.038, -5.3,     0,      0,     0,    0,   0,  128., 26.,      0,  2,   1000],
         [21,   1,   1.050,   0.,  400.,  115.1,  -50., 200.,   0,     0,   0,      0,  2,   1000],
         [22,   1,   1.050, 5.87,  300.,  -30.1,  -60.,  96.,   0,     0,   0,      0,  2,   1000],
         [23,   1,   1.050,  -4.,  660.,   129.,  125., 310.,   0,     0,   0,      0,  2,   1000],
         [24,   3,   0.984, -11.,     0,      0,     0,    0,   0,     0,   0,      0,  2,   0000]]

#DLIN
#          0      1     2      3       4      5       6       7       8       9    10      11       12 
#        (De)   (Pa)  NcEP  ( R% )  ( X% )  (Mvar)  (Tap)   (Tmn)   (Tmx)   (Phs) (Cn)    (Ce)   Ns(Cq)
D_LIN = [[ 1,    2,   1,      .26,   1.39,  46.11,      1,     0,     0,      0,  175.,  200.,    175.],
         [ 1,    3,   1,     5.46,  21.12,   5.72,      1,     0,     0,      0,  175.,  220.,    175.],
         [ 1,    5,   1,     2.18,   8.45,   2.29,      1,     0,     0,      0,  175.,  220.,    175.],
         [ 2,    4,   1,     3.28,  12.67,   3.43,      1,     0,     0,      0,  175.,  220.,    175.],
         [ 2,    6,   1,     4.97,   19.2,    5.2,      1,     0,     0,      0,  175.,  220.,    175.],
         [ 3,    9,   1,     3.08,   11.9,   3.22,      1,     0,     0,      0,  175.,  220.,    175.],
         [ 3,   24,   1,      .23,   8.39,      0,  1.015,   .95,   1.1,      0,  400.,  600.,  33400.],
         [ 4,    9,   1,     2.68,  10.37,   2.81,      1,     0,     0,      0,  175.,  220.,    175.],
         [ 5,   10,   1,     2.28,   8.83,   2.39,      1,     0,     0,      0,  175.,  220.,    175.],
         [ 6,   10,   1,     1.39,   6.05,  245.9,      1,     0,     0,      0,  175.,  200.,    175.],
         [ 7,    8,   1,     1.59,   6.14,   1.66,      1,     0,     0,      0,  175.,  220.,    175.],
         [ 8,    9,   1,     4.27,  16.51,   4.47,      1,     0,     0,      0,  175.,  220.,    175.],
         [ 8,   10,   1,     4.27,  16.51,   4.47,      1,     0,     0,      0,  175.,  220.,    175.],
         [ 9,   11,   1,      .23,   8.39,      0,   1.03,   .95,   1.1,      0,  400.,  600.,  33400.],
         [ 9,   12,   1,      .23,   8.39,      0,   1.03,   .95,   1.1,      0,  400.,  600.,  33400.],
         [10,   11,   1,      .23,   8.39,      0,  1.015,   .95,  1.05,      0,  400.,  600.,  33400.],
         [10,   12,   1,      .23,   8.39,      0,  1.015,   .95,  1.05,      0,  400.,  600.,  33400.],
         [11,   13,   1,      .61,   4.76,   9.99,      1,     0,     0,      0,  500.,  625.,    500.],
         [11,   14,   1,      .54,   4.18,   8.79,      1,     0,     0,      0,  500.,  625.,    500.],
         [12,   13,   1,      .61,   4.76,   9.99,      1,     0,     0,      0,  500.,  625.,    500.],
         [12,   23,   1,     1.24,   9.66,   20.3,      1,     0,     0,      0,  500.,  625.,    500.],
         [13,   23,   1,     1.11,   8.65,  18.18,      1,     0,     0,      0,  500.,  625.,    500.],
         [14,   16,   1,       .5,   3.89,   8.18,      1,     0,     0,      0,  500.,  625.,    500.],
         [15,   16,   1,      .22,   1.73,   3.64,      1,     0,     0,      0,  500.,  625.,    500.],
         [15,   21,   1,      .63,    4.9,   10.3,      1,     0,     0,      0,  500.,  625.,    500.],
         [15,   21,   2,      .63,    4.9,   10.3,      1,     0,     0,      0,  500.,  625.,    500.],
         [15,   24,   1,      .67,   5.19,  10.91,      1,     0,     0,      0,  500.,  625.,    500.],
         [16,   17,   1,      .33,   2.59,   5.45,      1,     0,     0,      0,  500.,  625.,    500.],
         [16,   19,   1,       .3,   2.31,   4.85,      1,     0,     0,      0,  500.,  625.,    500.],
         [17,   18,   1,      .18,   1.44,   3.03,      1,     0,     0,      0,  500.,  625.,    500.],
         [17,   22,   1,     1.35,  10.53,  22.12,      1,     0,     0,      0,  500.,  625.,    500.],
         [18,   21,   1,      .33,   2.59,   5.45,      1,     0,     0,      0,  500.,  625.,    500.],
         [18,   21,   2,      .33,   2.59,   5.45,      1,     0,     0,      0,  500.,  625.,    500.],
         [19,   20,   1,      .51,   3.96,   8.33,      1,     0,     0,      0,  500.,  625.,    500.],
         [19,   20,   2,      .51,   3.96,   8.33,      1,     0,     0,      0,  500.,  625.,    500.],
         [20,   23,   1,      .28,   2.16,   4.55,      1,     0,     0,      0,  500.,  625.,    500.],
         [20,   23,   2,      .28,   2.16,   4.55,      1,     0,     0,      0,  500.,  625.,    500.],
         [21,   22,   1,      .87,   6.78,  14.24,      1,     0,     0,      0,  500.,  625.,    500.]]

#%% Criação das listas a serem utilizadas no problema

nb = len(D_BAR) #número de barras
nl = len(D_LIN) #número de ramos

B_de = [] #identificar DE onde barra vem
B_para = [] #identificar PARA onde barra vai

r = [] #lista para os valores das resistência
x = [] #lista para os valores das reatâncias
z = [] #lista para os valores de impedância - z = r + jx
ysh_lin = [] #lista para os valores das admitâncias shunt
ysh_bar = [] #lista para os valores das admitâncias devido à compensação de reativo
a = [] #lista para os valores de tap dos transformadores

#%% Formação das variáveis a partir das listas

for i in range(nb):
    ysh_bar.append(complex(0, (1/100) * D_BAR[i][11])) #pega o valor da coluna 11 (Sh) para cada linha i e forma valores de admitância shunt devido à compensação de reativo

for i in range(nl):
    B_de.append(D_LIN[i][0]) #pega o valor da coluna 0 (DE) para cada linha i da matriz D_LIN
    B_para.append(D_LIN[i][1]) #pega o valor da coluna 1 (PARA) para cada linha i
    r.append((1/100) * D_LIN[i][3]) #pega o valor da coluna 3 (R%) para cada linha i - valores de resistência | se divide por 100 pois é dado em %
    x.append((1/100) * D_LIN[i][4]) #pega o valor da coluna 4 (X%) para cada linha i - valores de reatância
    z.append(complex(r[i], x[i])) #forma os valores de impedância a partir dos valores de r e x
    ysh_lin.append((1/(2*100)) * D_LIN[i][5]) #pega os valores da coluna 5 (Mvar) para cada linha i e forma o valor das susceptâncias shunt
                                                #Th pi contém metade da impedância em cada ramo shunt
    a.append(D_LIN[i][6]) #pega o valor da coluna 6 (Tap) para cada linha i e forma o valor de tap dos transformadores

#%% Cálculo das admitâncias

ykm = [0 for j in range(nl)] #inicialização de valores nulos para as admitâncias longitudinais

for k in range(nl):
    if r[k] != 0 and x[k] != 0:
        ykm[k] = 1/complex(r[k], x[k]) #cálculo das admitâncias longitudinais
    
    else:
        if r[k] == 0 and x[k] != 0:
            ykm[k] = 1/complex(0, x[k]) #cálculo das admitâncias quando a resistência é nula
    
        else:
            if x[k] == 0 and r[k] != 0:
                ykm[k] = 1/complex(r[k], 0) #cálculo das admitâncias quando a reatância é nula

            else:
                if r[k] == 0 and x[k] == 0:
                    ykm[k] = complex(0, 0) #cálculo das admitâncias quando ambas resistências e reatâncias são nulas

#%% Determinação da Y_barra

Y_barra = [[complex(0, 0) for j in range(nb)] for i in range(nb)] #inicialização da matriz com valores complexos nulos

for k in range(nb):
    Y_barra[k][k] += ysh_bar[k] #reatância da Y_barra devido à compensação de reativos

for k in range(nl):
    m = B_de[k] - 1 #criação de loop para barras DE
    n = B_para[k] - 1 #criação de loop para barras PARA

    Y_barra[m][m] += complex(0, ysh_lin[k]) #reatância da Y_barra devido à compensação de reativos
    Y_barra[m][m] += ykm[k] / (a[k]**2)
    Y_barra[n][n] += complex(0, ysh_lin[k]) #reatância da Y_barra devido à compensação de reativos
    Y_barra[n][n] += ykm[k]

m = [] #lista para identificação de barra DE
n = [] #lista para identificação de barra PARA

for k in range(nl):
    m = B_de[k] - 1 #criação de loop para barras DE
    n = B_para[k] - 1 #criação de loop para barras PARA

    Y_barra[m][n] -= (1/(a[k])) * ykm[k] #+ complex(0, ysh_lin[k]) #forma admitâncias série e susceptâncias shunt
    Y_barra[n][m] -= (1/(a[k])) * ykm[k]

G = np.real(Y_barra)
B = np.imag(Y_barra)

# print(np.round(np.real(Y_barra), 4))
# print('\n')
# print(np.round(np.imag(Y_barra), 4))

# np.savetxt('G.csv', np.round(G, 4), delimiter=';', fmt='%.4f')
# np.savetxt('B.csv', np.round(B, 4), delimiter=';', fmt='%.4f')

#================================================== AVALIAÇÃO 1 ===================================================
#Trimestre: 2023.1
#Disciplina: 210001 - Análise de Redes I
#Docente: João Alberto Passos Filho
#Discente: Esteban Vicente Aguilar Bojorge
#Enunciado: Desenvolver um programa de Fluxo de Potência em Python ou MatLab para resolver o sistema de 24 Barras 
#           do IEEE. Apresentar código desenvolvido e relatório técnico.
#==================================================================================================================

from cmath import *

#%% Início do método de Newton

#Dados das 

baseMVA = 100 #potência base

V = [] #vetor de tensões
Th = [] #vetor de ângulos
Pg = [] #vetor de P gerada
Qg = [] #vetor de Q gerada
Pd = [] #P demandado
Qd = [] #Q demandado
Pesp = [] #vetor de P demandada
Qesp = [] #vetor de Q demandada
tipoBAR = [] #tipo de barra: 1: PV, 2: VTh, 3: PQ 

for i in range(nb):
    Pg.append((D_BAR[i][4])/baseMVA) #pega o valor da coluna 4 (Pg) e forma valores de potência ativa gerada
    Qg.append((D_BAR[i][5])/baseMVA) #pega o valor da coluna 5 (Qg) e forma valores de potência reativa gerada
    Pd.append((D_BAR[i][9])/baseMVA) #pega o valor da coluna 9 (Pg) e forma valores de potência ativa demandada
    Qd.append((D_BAR[i][10])/baseMVA) #pega o valor da coluna 10 (Qg) e forma valores de potência reativa demandada
    tipoBAR.append((D_BAR[i][1])) #pega o valor da coluna 1 (Tipo) e forma o vetor de valores de tipo de barra

#%% Inicialização dos Dados

#Inicialização de tensões e ângulos

print('''
Escolha a opção de inicialização dos valores para o cálculo do fluxo de potência:
[1] Dados do ANAREDE
[2] Flat Start
''')

init = None
while init not in (1, 2):
    init = int(input('Digite o número da opção: '))

    if init == 1:
        for i in range(nb):
            V.append(float(D_BAR[i][2])) #pega o valor da coluna 2 (V) e forma matriz
            Th.append(float(D_BAR[i][3]) * (np.pi/180)) #pega o valor da coluna 3 (Angl) e forma matriz de ângulos

    elif init == 2:
        for i in range(nb):
            if tipoBAR[i] == 1:
                V.append(float(D_BAR[i][2])) #pega o valor da coluna 2 (V) e forma matriz
                Th.append(0) #pega o valor da coluna 3 (Angl) e forma matriz de ângulos
            elif tipoBAR[i] == 2:
                V.append(float(D_BAR[i][2])) #pega o valor da coluna 2 (V) e forma matriz
                Th.append(float(D_BAR[i][3]) * (np.pi/180)) #pega o valor da coluna 3 (Angl) e forma matriz de ângulos
            elif tipoBAR[i] == 3:
                V.append(1)
                Th.append(0)

    else:
        print("Escolha novamente")
#inicialização de potências ativas e reativas

ti_ybarra = time.time()

for k in range(nb):
    
    if tipoBAR[k] == 1:
        Pesp.append(Pg[k] - Pd[k])
        Qesp.append(0)
    
    elif tipoBAR[k] == 2:
        Pesp.append(0)
        Qesp.append(0)
    
    else:
        Pesp.append(Pg[k] - Pd[k])
        Qesp.append(Qg[k] - Qd[k])

Pesp = np.transpose(np.array([Pesp]))
Qesp = np.transpose(np.array([Qesp]))

get_indexes = lambda tipoBAR, tipoBARs: [i for (y, i) in zip(tipoBARs, range(len(tipoBARs))) if tipoBAR == y]
index_pv = get_indexes(1, tipoBAR)
index_pq = get_indexes(3, tipoBAR)
size_pv = len(index_pv)
size_pq = len(index_pq)

P = np.zeros((nb, 1))
Q = np.zeros((nb, 1))

#Cálculo de P e Q a partir do chute inicial
for i in range(nb):

    if tipoBAR[i] == 1:

        for k in range(nb):
            P[i][0] = P[i][0] + V[i] * V[k] * (G[i][k] * np.cos(Th[i] - Th[k]) + B[i][k] * np.sin(Th[i] - Th[k]))

        Q[i][0] = 0
    
    elif tipoBAR[i] == 2:
        P[i][0] = 0
        Q[i][0] = 0

    else:
        for k in range(nb):
            P[i][0] = P[i][0] + V[i] * V[k] * (G[i][k] * np.cos(Th[i] - Th[k]) + B[i][k] * np.sin(Th[i] - Th[k]))
            Q[i][0] = Q[i][0] + V[i] * V[k] * (G[i][k] * np.sin(Th[i] - Th[k]) - B[i][k] * np.cos(Th[i] - Th[k]))

dP = np.subtract(Pesp, P)
dQ = np.subtract(Qesp, Q)
dy = np.concatenate((dP, dQ), axis=0)

tol = np.max(np.abs(dy))
iter = 0

#%% Método de Newton

while tol > 0.00001:

    #%% Jacobiana
    #H - derivada parcial de P em relação a Th
    H = np.zeros((nb, nb))
    
    for i in range(nb):
        
        for k in range(nb):
            
            if i == k:
                
                for n in range(nb):

                    H[i][k] = H[i][k] + V[i] * V[n] * (-G[i][n] * np.sin(Th[i] - Th[n]) + B[i][n] * np.cos(Th[i] - Th[n]))
                
                H[i][k] = H[i][k] - (V[i]**2) * B[i][i]
            
            else:
                H[i][k] = V[i] * V[k] * (G[i][k] * np.sin(Th[i] - Th[k]) - B[i][k] * np.cos(Th[i] - Th[k]))
    
    #N - derivada parcial de P em relação a V
    N = np.zeros((nb, nb))

    for i in range(nb):

        for k in range(nb):

            if i == k:

                for n in range(nb):
                    N[i][k] = N[i][k] + V[n] * (G[i][n] * np.cos(Th[i] - Th[n]) + B[i][n] * np.sin(Th[i] - Th[n]))

                N[i][k] = N[i][k] + V[i] * G[i][i]
            
            else:
                N[i][k] = V[i] * (G[i][k] * np.cos(Th[i] - Th[k]) + B[i][k] * np.sin(Th[i] - Th[k]))

    #M - derivadas parciais de Q em relação a Th
    M = np.zeros((nb, nb))

    for i in range(nb):

        for k in range(nb):

            if i == k:
                
                for n in range(nb):
                    M[i][k] = M[i][k] + V[i] * V[n] * (G[i][n] * np.cos(Th[i] - Th[n]) + B[i][n] * np.sin(Th[i] - Th[n]))

                M[i][k] = M[i][k] - V[i]**2 * G[i][i]
            
            else:
                M[i][k] = V[i] * V[k] * (-G[i][k] * np.cos(Th[i] - Th[k]) - B[i][k] * np.sin(Th[i] - Th[k]))

    #L - derivadas parciais de Q em relação a V
    L = np.zeros((nb, nb))

    for i in range(nb):

        for k in range(nb):

            if i == k:
                for n in range(nb):
                    L[i][k] = L[i][k] + V[n] * (G[i][n] * np.sin(Th[i] - Th[n]) - B[i][n] * np.cos(Th[i] - Th[n]))

                L[i][k] = L[i][k] - V[i] * B[i][i]
            
            else:
                L[i][k] = V[i] * (G[i][k] * np.sin(Th[i] - Th[k]) - B[i][k] * np.cos(Th[i] - Th[k]))

    #formação da Jacobiana
    J1 = np.concatenate((H, N), axis=1) #juntar matrizes de P
    J2 = np.concatenate((M, L), axis=1) #juntar matrizes de Q
    J = np.concatenate((J1, J2), axis=0)

    for k in range(nb):
        
        if tipoBAR[k] == 1:
            J[k+nb][k+nb] = 1e100
        
        elif tipoBAR[k] == 2:
            J[k][k] = 1e100
            J[k+nb][k+nb] = 1e100

    X = np.linalg.solve(J, dy)
    dTh = X[:nb] # Residuos de Th
    dV = X[nb:] # Residuos de V

    # Atualização do vetor solução:
    for i in range(nb):
        Th[i] = dTh[i] + Th[i]

    for i in range(nb):
        V[i] = dV[i] + V[i]
    
    dy = np.zeros((2*nb, 1))

    P = np.zeros((nb, 1))
    Q = np.zeros((nb, 1))

    #Cálculo de P e Q
    for i in range(nb):

        if tipoBAR[i] == 1:

            for k in range(nb):
                P[i] = P[i] + V[i] * V[k] * (G[i][k] * np.cos(Th[i] - Th[k]) + B[i][k] * np.sin(Th[i] - Th[k]))

            Q[i] = 0
        
        elif tipoBAR[i] == 2:
            P[i] = 0
            Q[i] = 0

        else:
            for k in range(nb):
                P[i] = P[i] + V[i] * V[k] * (G[i][k] * np.cos(Th[i] - Th[k]) + B[i][k] * np.sin(Th[i] - Th[k]))
                Q[i] = Q[i] + V[i] * V[k] * (G[i][k] * np.sin(Th[i] - Th[k]) - B[i][k] * np.cos(Th[i] - Th[k]))

    dP = np.subtract(Pesp, P)
    dQ = np.subtract(Qesp, Q)
    dy = np.concatenate((dP, dQ), axis=0)

    iter = iter + 1 # próxima iteração
    tol = np.max(np.abs(dy)) # cálculo de tolerância

#%% Cálculo das Potências em barras VTh e 
for i in range(nb):
    for k in range(nb):
        if tipoBAR[i] == 1:
            Q[i] = Q[i] + V[i] * V[k] * (G[i][k] * np.sin(Th[i] - Th[k]) - B[i][k] * np.cos(Th[i] - Th[k]))
        elif tipoBAR[i] == 2:
            P[i] = P[i] + V[i] * V[k] * (G[i][k] * np.cos(Th[i] - Th[k]) + B[i][k] * np.sin(Th[i] - Th[k]))
            Q[i] = Q[i] + V[i] * V[k] * (G[i][k] * np.sin(Th[i] - Th[k]) - B[i][k] * np.cos(Th[i] - Th[k]))

V = np.array(V)
Angl = np.array(Th)*180/np.pi
P = np.array(P * baseMVA)
Q = np.array(Q * baseMVA)

# print('As tensões por barra, em p.u., são:', '\n', str(V).replace('[', '').replace(']', ''), '\n')
# print('Os ângulos por barra, em graus, são:', '\n', str(Angl).replace('[', '').replace(']', ''), '\n')

#%% Fluxos de potências nas linhas
P_km = np.zeros((nl,))
Q_km = np.zeros((nl,))

for k in range(nl):
    f = B_de[k]
    t = B_para[k]
    akm = a[k]
    gkm = G[f-1,t-1]
    bkm = B[f-1,t-1]
    bskm = 0
    thkm = Th[f-1] - Th[t-1]
    phikm = 0
    Vk = V[f-1]
    Vm = V[t-1]
    Pkm = (akm * Vk)**2 * gkm - akm * Vk * Vm * gkm * np.cos(thkm + phikm) - akm * Vk * Vm * bkm * np.sin(thkm + phikm)
    Qkm = -(akm * Vk)**2 * (bkm + bskm) + akm * Vk * Vm * bkm * np.cos(thkm + phikm) - akm * Vk * Vm * gkm * np.sin(thkm + phikm)
    P_km[k] = Pkm
    Q_km[k] = Qkm

P_km = P_km * baseMVA
Q_km = Q_km * baseMVA

#%% Resultados
tf_ybarra = time.time()
print('Tempo de Execução:', round(tf_ybarra - ti_ybarra, 5), 's')
print('Convergência em:', iter, 'iterações', '\n')

#%% Criação de gráficos
# x_pos = np.arange(nb)
# y = V

# plt.plot(x_pos, V, label='Tensão', linestyle='-', color='#5979f2')
# plt.xlabel('Barras do Sistema')
# plt.ylabel('Tensão nas barras do sistema IEEE 24 Barras (p.u.)')
# plt.legend()
# plt.savefig('tensao24barras.pdf')
# plt.show()

#%% Visualização tabular

#Relatório de Barras:

for i in range(nb):
    if tipoBAR[i] == 1:
        tipoBAR[i] = 'PV'
    elif tipoBAR[i] == 2:
        tipoBAR[i] = 'V\u03B8'
    elif tipoBAR[i] == 3:
        tipoBAR[i] = 'PQ'
    else:
        break

df_b = pd.DataFrame(list(zip(tipoBAR, np.round(V, 4), np.round(Angl, 4), np.round(P, 4), np.round(Q, 4))), columns=['Tipo', 'Tensão (p.u.)', 'Ângulo (º)', 'Potência (MW)', 'Potência (Mvar)']) #cria dataframe no formato pandas
#df_b = df_b.apply(lambda x: x.replace('[','').replace(']','')) #remove '[' e ']'
df_b = df_b.astype({'Tensão (p.u.)':'float'}) #remove '[' e ']'
df_b = df_b.astype({'Ângulo (º)':'float'})
df_b = df_b.astype({'Potência (MW)':'float'})
df_b = df_b.astype({'Potência (Mvar)':'float'})

df_b.index = np.arange(1, len(df_b) + 1) #aumenta o index em 1, para condizer com número da barra
df_b.index.name = 'Barra' #dá nome ao index
df_b.to_csv('RBAR.csv', sep=';', encoding='utf-8-sig', decimal=',') #salva o index com nome de ANAREDE

print(df_b, '\n')

#Relatório de Linhas

df_l = pd.DataFrame(list(zip(B_de, B_para, P_km, Q_km)), columns=['De barra', 'Para barra', 'P_km',  'Q_km'])

df_l.index = np.arange(1, len(df_l) + 1) #aumenta o index em 1, para condizer com número da linha
df_l.index.name = 'Linha' #dá nome ao index
df_l.to_csv('RLIN.csv', sep=';', encoding='utf-8-sig', decimal=',') #salva o index com nome de ANAREDE

print(df_l)
