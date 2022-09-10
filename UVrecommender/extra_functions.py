import numpy as np
import numpy.random as nprnd
import math
import random
import os


def RMSE(M, Mhat, n, p):
    """Calcola RMSE tra le matrici M e Mhat"""
    
    res = M - Mhat
    num = 0
    den = 0
    
    for i in range(n):
        for j in range(p):
           if not math.isnan(res[i,j]):
               den += 1
               num += res[i,j]**2
    
    return np.sqrt(num/den)



def zero_nan_replace(A):
    """Data una matrice, ne restituisce una copia
    dove i Nan sono scambiati con 0"""
    
    n = np.shape(A)[0]
    p = np.shape(A)[1]
    B = A
    
    for i in range(n):
        for j in range(p):
            if math.isnan(A[i,j]):
                B[i,j] = 0
    
    return B



def evaluate_classification(A,Ahat):
    """Data una matrice e una sua stima restituisce 
    la percentuale di corretta classificazione 
    (considera corrette anche le stime >5 dove il vero 
     valore e' 5 e le stime <1 dove il vero valore e' 1)"""
    
    n = np.shape(A)[0]
    p = np.shape(A)[1]
    m=0
    
    for elem in A.flat:
        if not math.isnan(elem):
            m += 1
    
    class_good = 0
    for i in range(n):
        for j in range(p):
            if A[i,j] == 1 and round(Ahat[i,j]) < 1:
                class_good += 1
            if A[i,j] == round(Ahat[i,j]):
                class_good += 1
            if A[i,j] == 5 and round(Ahat[i,j]) > 5:
                class_good += 1
    
    return 100*class_good/m



def test_matrix(n, p):
    """Crea una matrice di prova n*p 
    avente circa 25% di Nan per riga"""
    
    mat = np.zeros([n,p])
    for i in range(n):
        for j in range(p):
            mat[i,j] = nprnd.randint(1,6)
    
    for i in range(n):
            nan_index = random.sample(list(range(p)),int(0.25*p))
            for j in nan_index:
                mat[i,j] = float('nan')            
    
    return mat 



def create_observed_couple(M):
    """Salva in una lista gli indici delle entrate non
    vuote di una matrice sparsa"""
    
    n = np.shape(M)[0]  
    p = np.shape(M)[1]
    S=[]
    obs=0
    
    for i in range(n):
        for j in range(p):
            if not math.isnan(M[i,j]):
                S.append((i,j))
                obs+=1
    
    return S, obs



def matrix_mean(M):
    """Calcola la media delle entrate di una matrice"""
    
    nrow=M.shape[0]
    ncol=M.shape[1]
    num=0
    den=0
    
    for i in range(nrow):
        for j in range(ncol):
            if not math.isnan(M[i,j]):
                num += M[i,j]
                den += 1
    
    return num/den



def matrix_perturbation(A, mean=0, scale=1):
    """Perturba le entrate di una matrice secondo
    una distribuzione normale"""
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            t=A[i,j]
            A[i,j] = (t + nprnd.normal(loc=mean, scale=scale))
       
            
       
def matrix_perturbation_unif(A, inf = -1, sup = 1):
    """Perturba le entrate di una matrice secondo
    una distribuzione uniforme"""
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            t=A[i,j]
            A[i,j]= (t + nprnd.uniform(low=inf, high=sup))
            
            
def matrix_mean_notNan_byrow(M, n, p):
    """Data una matrice M n*p restituisce il vettore di 
    lunghezza n contenente le medie di riga, escludendo Nan"""
    
    mean_byrow = list()
    
    for i in range(n):
        mean_i = 0
        n_notnans = 0
        for j in range(p):
            if not math.isnan(M[i,j]):
                mean_i += M[i,j]
                n_notnans +=1
        mean_i = mean_i/n_notnans
        mean_byrow.append(mean_i)
    
    return mean_byrow



def matrix_mean_notNan_bycolumn(M, n, p):
    """Data una matrice M n*p restituisce il vettore di 
    lunghezza p contenente le medie di colonna, escludendo Nan"""
    
    mean_bycolumn = list()
    
    for j in range(p):
        mean_j = 0
        n_notnans = 0
        for i in range(n):
            if not math.isnan(M[i,j]):
                mean_j += M[i,j]
                n_notnans +=1
        mean_j = mean_j/n_notnans
        mean_bycolumn.append(mean_j)
    
    return mean_bycolumn  



def initialization_meanrow(M, n, p, d):
    """Data una matrice M(n*p), restituisce U(n*d) e V(d*p) tali 
    che U*V(n*p) ha elementi sulla stessa riga uguali fra loro e
    pari alla corrispondente media di riga di M, escludendo Nan"""
    
    U = np.zeros((n, d)) 
    V = np.zeros((d, p))
    
    means_byline = matrix_mean_notNan_byrow(M, n, p)
    
    for i in range(n):
        U[i] = means_byline[i]/(d*matrix_mean(M))
    
    for q in range(d):
        for j in range(p):
            V[q,j] = matrix_mean(M)
    
    return U, V



def initialization_meancolumn(M, n, p, d):
    """Data una matrice M(n*p), restituisce U(n*d) e V(d*p) tali 
    che U*V(n*p) ha elementi sulla stessa colonna uguali fra loro e 
    pari alla corrispondente media di colonna di M, escludendo Nan"""
    
    U = np.zeros((n, d)) 
    V = np.zeros((d, p))
    
    means_bycolumn = matrix_mean_notNan_bycolumn(M, n, p)
    
    for i in range(n):
        for q in range(d):
            U[i,q]= matrix_mean(M)
    
    for j in range(p):
            V[:,j] = means_bycolumn[j]/(d*matrix_mean(M))
    
    return U, V                


def initialization(M, d , n, p, init = "ones"):
    
    if init == "mean":
        U = np.zeros((n, d)) + np.sqrt(matrix_mean(M)/d)
        V = np.zeros((d, p)) + np.sqrt(matrix_mean(M)/d)

    elif init == "meanrow":
        U, V = initialization_meanrow(M, n, p, d)
        
    elif init == "meancol":
        U, V = initialization_meancolumn(M, n, p, d)

    else:
        U = np.ones((n, d))
        V = np.ones((d, p))

    return U, V



def perturbation(U, V, nperturb, method="n"):
    """Inizializza le matrici U e V come specificato 
    dal parametro init, restituisce z copie di tali 
    matrici perturbate come specificato dal parametro
    method"""
    
    n = U.shape[0]
    p = V.shape[1]
    d = U.shape[1]
    
    Uz = np.zeros((n, d, nperturb))
    Vz = np.zeros((d, p, nperturb))
        
    for z in range(nperturb):
        Uz[:,:,z] = U
        Vz[:,:,z] = V
        
        if method == "u":
            matrix_perturbation_unif(Uz[:,:,z])
            matrix_perturbation_unif(Vz[:,:,z])
        else:
            matrix_perturbation(Uz[:,:,z])   
            matrix_perturbation(Vz[:,:,z])
   
    return (Uz, Vz) 

 

def matrix_to_text(utility_hat, utility, begin, end, filenum):
    """Converte le righe comprese tra begin e end di una matrice 
    utility_hat in un file di testo in formato i,j,M[i,j], dandogli 
    nome MatrixN.txt con N pari a filenum e ignorando valori presenti 
    nella matrice utility iniziale"""
    
    cwd=os.getcwd()
    file_path=cwd+"\Matrix{}.txt".format(str(filenum))
    file_obj=open(file_path,"w")
    
    p=utility_hat.shape[1]
    for i in range(begin,end):
        for j in range(p):
            if math.isnan(utility[i,j]):
                file_obj.write("{},{},{}\n".format(i,j,utility_hat[i,j]))
    file_obj.close()
    
    return 