# -*- coding: utf-8 -*-

import time
import datetime as dt
import numpy as np

import GD
import SGD
import SGD2

import user_song as us
#import extra_functions as ef



"""File di testo in cui riportare i risultati dell'analisi"""
path = "analisi1.txt"
f = open(path, "w") 
f.write("""Stima di valori mancanti all'interno della matrice di utilita' corrispondente 
al dataset user_song, mediante l'utilizzo di tre diversi algoritmi di discesa del 
gradiente e in base a diverse combinazioni di parametri opzionali degli algoritmi\n""")
        
#utility = ef.test_matrix(5, 5)
utility, dati = us.create_utility_matrix(True)

n = np.shape(utility)[0]
p = np.shape(utility)[1]
m = np.shape(dati)[0]
nan_perc = np.round(100*(1 - m/(n*p)),2)   
    
print("\nMatrice di utilità (n*p) con n = {} users e p = {} canzoni\nPercentuale di rating non assegnati : {} %\n".format(n,p, nan_perc))
f.write("\nMatrice di utilità (n*p) con n = {} users e p = {} canzoni\nPercentuale di rating non assegnati : {} %\n".format(n,p, nan_perc))


"""Numero di dimensioni latenti pari a 3 individuato euristicamente utilizzando il modulo search_for_d.py
Numero di perturbazioni pari a 2 funzionale a dimensione dataset
Valori di eta per le tre procedure individuati euristicamente"""
d = 2
nperturb = 3

etaGD = 0.0001
etaSGD = 0.0001
etaSGD2 = 0.0001

f.write("""\nNumero di dimensioni latenti pari a 2 individuato euristicamente utilizzando il 
modulo search_for_d.py; numero di perturbazioni pari a 3 funzionale a dimensione 
dataset, valori di eta per le tre procedure individuati euristicamente""")



"""Analisi 1"""
f.write("""\n\n\nAnalisi 1: inizializzazione di U e V tale che la matrice stimata iniziale contenga i 
valori medi della matrice di utilità, senza perturbazioni di U e V inizializzate\n""")
f.write("Algoritmo\tRMSE\t\tclassificazione\t\titerazioni\ttempo totale\n")
time.sleep(1)
t1 = dt.datetime.now()
U, V, rmse, num_iter, perc_corretti = GD.gradient_descent_mat(utility, d = d, eta = etaGD, perturb = False, init = "mean")
t2 = dt.datetime.now()
delta = t2 - t1
f.write("GD\t\t{}\t\t{}%\t\t\t{}\t\t{}\n".format(np.round(rmse,5), np.round(perc_corretti,2), num_iter, delta)) 
time.sleep(1)
t1 = dt.datetime.now()
U, V, rmse, num_iter, perc_corretti = SGD.stochastic_gradient_descent_mat(utility, d = d, eta = etaSGD, perturb = False, init = "mean")
t2 = dt.datetime.now()
delta = t2 - t1
f.write("SGD\t\t{}\t\t{}%\t\t\t{}\t\t{}\n".format(np.round(rmse,5), np.round(perc_corretti,2), num_iter, delta)) 
time.sleep(1)
t1 = dt.datetime.now()
U, V, rmse, num_iter, perc_corretti = SGD2.stochastic_gradient_descent_mat2(utility, d = d, eta = etaSGD2, perturb = False, init = "mean")
t2 = dt.datetime.now()
delta = t2 - t1
f.write("SGD2\t\t{}\t\t{}%\t\t\t{}\t\t{}\n".format(np.round(rmse,5), np.round(perc_corretti,2), num_iter, delta)) 



"""Analisi 1a"""
f.write("""\nAnalisi 1a: inizializzazione di U e V tale che la matrice stimata iniziale contenga i
valori medi della matrice di utilità, con 2 perturbazioni (secondo una distribuzione 
normale standard) delle matrici U e V inizializzate\n""")
f.write("Algoritmo\tRMSE\t\tclassificazione\t\titerazioni\ttempo totale\n")
time.sleep(1)
t1 = dt.datetime.now()
U, V, rmse, num_iter, perc_corretti = GD.gradient_descent_mat(utility, d = d, eta = etaGD, perturb = True, nperturb = nperturb, method = "n", init = "mean")
t2 = dt.datetime.now()
delta = t2 - t1
f.write("GD\t\t{}\t\t{}%\t\t\t{}\t\t{}\n".format(np.round(rmse,5), np.round(perc_corretti,2), num_iter, delta)) 
time.sleep(1)
t1 = dt.datetime.now()
U, V, rmse, num_iter, perc_corretti = SGD.stochastic_gradient_descent_mat(utility, d = d, eta = etaSGD, perturb = True, nperturb = nperturb, method = "n", init = "mean")
t2 = dt.datetime.now()
delta = t2 - t1
f.write("SGD\t\t{}\t\t{}%\t\t\t{}\t\t{}\n".format(np.round(rmse,5), np.round(perc_corretti,2), num_iter, delta)) 
time.sleep(1)
t1 = dt.datetime.now()
U, V, rmse, num_iter, perc_corretti = SGD2.stochastic_gradient_descent_mat2(utility, d = d, eta = etaSGD2, perturb = True, nperturb = nperturb, method = "n", init = "mean")
t2 = dt.datetime.now()
delta = t2 - t1
f.write("SGD2\t\t{}\t\t{}%\t\t\t{}\t\t{}\n".format(np.round(rmse,5), np.round(perc_corretti,2), num_iter, delta)) 



"""Analisi 1b"""
f.write("""\nAnalisi 1b: inizializzazione di U e V tale che la matrice stimata iniziale contenga i 
valori medi della matrice di utilità, con 2 perturbazioni (secondo una distribuzione 
uniforme tra -1 e 1) delle matrici U e V inizializzate\n""")
f.write("Algoritmo\tRMSE\t\tclassificazione\t\titerazioni\ttempo totale\n")
time.sleep(1)
t1 = dt.datetime.now()
U, V, rmse, num_iter, perc_corretti = GD.gradient_descent_mat(utility, d = d, eta = etaGD, perturb = True, nperturb = nperturb, method = "u", init = "mean")
t2 = dt.datetime.now()
delta = t2 - t1
f.write("GD\t\t{}\t\t{}%\t\t\t{}\t\t{}\n".format(np.round(rmse,5), np.round(perc_corretti,2), num_iter, delta)) 
time.sleep(1)
t1 = dt.datetime.now()
U, V, rmse, num_iter, perc_corretti = SGD.stochastic_gradient_descent_mat(utility, d = d, eta = etaSGD, perturb = True, nperturb = nperturb, method = "u", init = "mean")
t2 = dt.datetime.now()
delta = t2 - t1
f.write("SGD\t\t{}\t\t{}%\t\t\t{}\t\t{}\n".format(np.round(rmse,5), np.round(perc_corretti,2), num_iter, delta)) 
time.sleep(1)
t1 = dt.datetime.now()
U, V, rmse, num_iter, perc_corretti = SGD2.stochastic_gradient_descent_mat2(utility, d = d, eta = etaSGD2, perturb = True, nperturb = nperturb, method = "u", init = "mean")
t2 = dt.datetime.now()
delta = t2 - t1
f.write("SGD2\t\t{}\t\t{}%\t\t\t{}\t\t{}\n".format(np.round(rmse,5), np.round(perc_corretti,2), num_iter, delta)) 



f.close()
