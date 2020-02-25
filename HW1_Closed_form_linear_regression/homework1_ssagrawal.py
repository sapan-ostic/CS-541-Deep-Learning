import numpy as np

def problem_a (A, B):
    return A + B

def problem_b (A, B, C):
    return np.dot(A,B)-C

def problem_c (A, B, C):
    return A*B+C.transpose()

def problem_d (x, y):
    return np.dot(x.transpose(),y)

def problem_e (A):
    return np.zeros(A.shape)

def problem_f (A, x):
    return np.linalg.solve(A,x)

def problem_g (A, x):
    return np.linalg.solve(A.transpose(),x.transpose()).transpose()

def problem_h (A, alpha):
    return A+alpha*np.eye(A.shape[0])

def problem_i (A, i, j):
    return A[i][j]

def problem_j (A, i):
    return np.sum(A[i,::2])

def problem_k (A, c, d):
    return np.mean(A*(A<=d)*(A>=c))

def problem_l (A, k):
    w,v = np.linalg.eig(A)
    sort_id = np.argsort(w)[-k:] 
    return print(v[:, sort_id[::-1]]) 

def problem_m (x, k, m, s):
    n = len(x)
    z = np.ones([n])
    mean = x + np.dot(m,z)
    cov = s*np.eye(n)
    return np.random.multivariate_normal(mean, cov, size=k).transpose()

def problem_n (A):
    n = A.shape[0]
    ind = np.random.permutation(n)
    return A[ind]

def linear_regression (X_tr, y_tr):
    n,m = X_tr.shape
    X = np.zeros([1,m])
    for i in range(n):
        X += (X_tr[i,:]) * y_tr[i]
    A = np.dot(X_tr.transpose(), X_tr)
    w = np.linalg.solve(A,X.transpose())
    return w
    
def train_age_regressor ():
    # Load data
    X_tr = np.load("age_regression_Xtr.npy")
    n = X_tr.shape[0]
    X_tr = X_tr.reshape((n,-1))
    ytr = np.load("age_regression_ytr.npy")
    
    X_te = np.load("age_regression_Xte.npy")
    m = X_te.shape[0]
    X_te = X_te.reshape((m,-1))
    
    yte = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)
    return w

def report_fMSE():
	w = train_age_regressor()
	X_tr = np.load("age_regression_Xtr.npy")
	n = X_tr.shape[0]
	X_tr = X_tr.reshape((n,-1))
	ytr = np.load("age_regression_ytr.npy")

	X_te = np.load("age_regression_Xte.npy")
	m = X_te.shape[0]
	X_te = X_te.reshape((m,-1))
	yte = np.load("age_regression_yte.npy")

	fMSE = 0
	for i in range(n):
	    fMSE += (1/(2*n))*(np.dot(X_tr[i,:],w)- ytr[i])*(np.dot(X_tr[i,:],w)- ytr[i]) 
	    
	print("Training Data fMSE = ", fMSE[0])

	fMSE = 0
	for i in range(m):
	    fMSE += (1/(2*m))*(np.dot(X_te[i,:],w)- yte[i])*(np.dot(X_te[i,:],w)- yte[i]) 
	    
	print("Testing Data fMSE = ", fMSE[0])

report_fMSE()
