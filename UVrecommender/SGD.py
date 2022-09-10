import numpy as np
import numpy.random as nprnd
import argparse

import extra_functions as ef

epsilon = -0.0007
MAX_ITER = 1500

"""M(n*p)[i,j]   U(n*d)[i,q]   V(d*p)[q,j]"""


def stochastic_gradient_descent_mat(M, d, eta=0.0001, onlyUV = False, perturb = False, nperturb = 3, method="n", init="mean"):
    n = np.shape(M)[0]  
    p = np.shape(M)[1]
    s, nobs = ef.create_observed_couple(M)

    if perturb == False:    

        U, V = ef.initialization(M, d, n, p, init = init)
    
        rmse=list()
        rmse.append(ef.RMSE(M, np.dot(U, V), n, p))
        print("\nRMSE iniziale = {:.2f}".format(rmse[0]))
    
        CONVERGED = False
        num_iter = 0
        while not CONVERGED and num_iter < MAX_ITER:
            num_iter += 1
            E = M - np.dot(U,V)
                        
            nprnd.shuffle(s)  
            for i in s:
                u_i_up = U[i[0]] + eta*E[i]*V[:,i[1]]
                v_j_up = V[:,i[1]] + eta*E[i]*U[i[0]]
                U[i[0]] = u_i_up
                V[:,i[1]] = v_j_up
        
            rmse.append(ef.RMSE(M, np.dot(U, V), n, p))
            if (rmse[num_iter] - rmse[num_iter-1]) <= 0 and (rmse[num_iter] - rmse[num_iter-1]) > epsilon:
                CONVERGED = True 

            print("iterazione: {} | rmse={}".format(num_iter, rmse[num_iter]))

        if not CONVERGED:
            print("non converge dopo {} iterazioni".format(num_iter))
        else:
            print("convergenza dopo {} iterazioni [rmse={}]".format(
                    num_iter, rmse[num_iter]))

        perc_corretti = ef.evaluate_classification(M, np.dot(U,V))    
    
        if onlyUV == True:
            return (U, V)
        else:
            return (U, V, rmse[num_iter], num_iter, perc_corretti)

    else:
        U, V = ef.initialization(M, d, n, p, init = init)
        U, V = ef.perturbation(U, V, nperturb, method = method)
        avg_num_iter = 0
        
        for z in range(nperturb):
            rmse=list()
            rmse.append(ef.RMSE(M, np.dot(U[:,:,z], V[:,:,z]), n, p))
            print("\nRMSE iniziale ( perturbazione {}) = {:.2f}".format(z+1,rmse[0]))
        
            CONVERGED = False
            num_iter = 0
            while not CONVERGED and num_iter < MAX_ITER:
                num_iter += 1
                E = M - np.dot(U[:,:,z],V[:,:,z])
                
                nprnd.shuffle(s)  
                for i in s:
                    u_i_up = U[i[0],:,z] + eta*E[i]*V[:,i[1],z]
                    v_j_up = V[:,i[1],z] + eta*E[i]*U[i[0],:,z]
                    U[i[0],:,z] = u_i_up
                    V[:,i[1],z] = v_j_up
        
                rmse.append(ef.RMSE(M, np.dot(U[:,:,z], V[:,:,z]), n, p))
                if (rmse[num_iter] - rmse[num_iter-1]) <= 0 and (rmse[num_iter] - rmse[num_iter-1]) > epsilon:
                    CONVERGED = True  
                
                print("(perturbazione {}) iterazione: {} | rmse={}".format(z+1,num_iter, rmse[num_iter]))

            if not CONVERGED:
                print("(perturbazione {}) non converge dopo {} iterazioni".format(z+1,num_iter))
            else:
                print("(perturbazione {}) convergenza dopo {} iterazioni [rmse={}]".format(
                    z+1,num_iter, rmse[num_iter]))
                
            avg_num_iter += num_iter

        avg_num_iter /= nperturb   
        avg_num_iter = np.round(avg_num_iter,2)                         
        Ufin = np.apply_along_axis(np.mean, 2, U)
        Vfin = np.apply_along_axis(np.mean, 2, V)            
        perc_corretti = ef.evaluate_classification(M, np.dot(Ufin, Vfin))
        rmse_final = ef.RMSE(M, np.dot(Ufin, Vfin), n, p)
        
        if onlyUV == True:
            return (Ufin, Vfin)
        else:
            return (Ufin, Vfin, rmse_final, avg_num_iter, perc_corretti)    



def main():
    
    parser = argparse.ArgumentParser(
        description="Calcola la decomposizione UV di una data matrice con metodo Gradient Descent.")
    parser.add_argument(
        "--path", help="percorso al file contenente la matrice di prova 5*5", default = "L.txt")
    parser.add_argument(
        "-d", type=int, help="numero di dimensioni latenti", required=True)
    parser.add_argument(
        "-eta", type=float, help="learning rate dell'algoritmo GD (default = 0.005)", default = 0.005)
    parser.add_argument(
        "-n", type=int, help="numero di righe della matrice di prova casuale (default = 10)", default = 10)
    parser.add_argument(
        "-p", type=int, help="numero di colonne della matrice di prova casuale (default = 20)", default = 20)
    parser.add_argument(
        "-perturb", type=bool, help="si vuole effettuare una perturbazione delle matrici U e V iniziali ? (default = False)", default = False)
    parser.add_argument(
        "-nperturb", type=int, help="numero di perturbazioni se nperturb==True (default = 3)", default = 3)
    parser.add_argument(
        "-init", help="metodo di inizializzazione delle matrici U e V (default='ones',alternativa='mean')", default = "ones")
    parser.add_argument(
        "-method", help="metodo di perturbazione delle matrici U e V (default='n', alternativa='u')", default = "n")
    
    
    args = vars(parser.parse_args())  
    
    d = args["d"]
    eta = args["eta"]
    n = args["n"]
    p = args["p"]
    perturb = args["perturb"]
    nperturb = args["nperturb"]
    init = args["init"]
    method = args["method"]
    
    with open(args["path"], 'r', encoding='utf-8-sig') as f:
        M = np.genfromtxt(f, dtype=float, delimiter=',')
        
    print("\nMatrice di prova 5*5:\n")    
    print("M = ")
    print(M)
    
    if perturb == False:
        (U, V, rmse, num_iter, perc_corretti) = stochastic_gradient_descent_mat(M, d, eta)	

        print("\nU = ")
        print(np.around(U, decimals=2))
        print("\nV = ")
        print(np.around(V, decimals=2))

        print("\nM = ")
        print(M)    
        print("\nM.hat = ")
        print(np.around(np.dot(U,V), decimals=2))
        print("\nRMSE = {}\nNumero medio di iterazioni dell'algoritmo : {}\nPercentuale di rating classificati correttamente pari a : {} %\n\n".format(rmse, num_iter, np.round(perc_corretti,2)))
   
    else:
        (U, V ,rmse, avg_num_iter, perc_corretti) = stochastic_gradient_descent_mat(M, d, eta, onlyUV = False, perturb = perturb, nperturb = nperturb, init = init, method = method)
   
        print("\nU = ")
        print(np.around(U, decimals=2))
        print("\nV = ")
        print(np.around(V, decimals=2))

        print("\nM = ")
        print(M)    
        print("\nM.hat = ")
        print(np.around(np.dot(U,V), decimals=2))
        print("\nRMSE = {}\nNumero medio di iterazioni dell'algoritmo : {}\nPercentuale di rating classificati correttamente pari a : {} %\n\n".format(rmse, np.round(avg_num_iter,2), np.round(perc_corretti,2)))
       
   
    
    print("\nMatrice di prova casuale {}*{}:\n".format(n,p))
    M = ef.test_matrix(n, p)
    print("M = ")
    print(M)
        
    if perturb == False:
        (U, V, rmse, num_iter, perc_corretti) = stochastic_gradient_descent_mat(M,d,eta)	

        print("\nU = ")
        print(np.around(U, decimals=2))
        print("\nV = ")
        print(np.around(V, decimals=2))

        print("\nM = ")
        print(M)    
        print("\nM.hat = ")
        print(np.around(np.dot(U,V), decimals=2))
        print("\nRMSE = {}\nNumero medio di iterazioni dell'algoritmo : {}\nPercentuale di rating classificati correttamente pari a : {} %\n\n".format(rmse, num_iter, np.round(perc_corretti,2)))
   
    else:
        (U, V,rmse, avg_num_iter, percorretti) = stochastic_gradient_descent_mat(M, d, eta, onlyUV = False, perturb = perturb, nperturb = nperturb, init = init, method = method)
   
        print("\nU = ")
        print(np.around(U, decimals=2))
        print("\nV = ")
        print(np.around(V, decimals=2))

        print("\nM = ")
        print(M)    
        print("\nM.hat = ")
        print(np.around(np.dot(U,V), decimals=2))
        print("\nRMSE = {}\nNumero medio di iterazioni dell'algoritmo : {}\nPercentuale di rating classificati correttamente pari a : {} %\n\n".format(rmse, np.round(avg_num_iter,2), np.round(perc_corretti,2)))
       
   
if __name__ == '__main__':
    main()          