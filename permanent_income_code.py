import numpy as np
import matplotlib.pyplot as plt




# =============================================================================
#  2.3: Impulse response -- Y
# =============================================================================
def income_path(T=2000, S=2, sigma1=0.14364, sigma2=0.20615):
    """
    Time path of log income given shock sequence.
    
	Input
	=======
	T: the time horizon, default 2000
	S: Impulse date, default 2
	sigma1: permanent shock, default 0.14364
	sigma2: transitory shock, default 0.20615

    Output
    =======
    Y1: the impulse response path of income regarding the permanent shock
    Y2: the impulse response path of income regarding the transitory shock
    """
    w = np.zeros(T+S)
    X1 = np.zeros(T+S)
    X2 = np.zeros(T+S)
    Y1 = np.zeros(T+S) 
    Y2 = np.zeros(T+S)  
    w[S] = 1
    
    for t in range(1, T+S-1):
        X1[t+1] = 0.704 * X1[t] + sigma1 * w[t+1]
        Y1[t+1] = Y1[t] + X1[t+1]
        X2[t+1] = X2[t] - 0.154 * X2[t-1] + sigma2 * w[t+1]
        Y2[t+1] = X2[t+1]
    return Y1, Y2


# =============================================================================
# 2.3： Impulse response -- C + Y
# =============================================================================
def consumption_income_ratio_path(M, T=2000, S=2, rho=0.00663, nu=0.00373,
								  sigma1=0.14364, sigma2=0.20615):
    """
    Time path of log consumption-income ratio given shock sequence
     
	Input
	=======
	M: the solution matrix of expressing the non-financial income contribution 
	   to the log consumption-income ratio with respect to the states, i.e.,
	   M = \lambda D'(I = \lambda A)^{-1}
	rho: the asset return, default 0.00663
	nu: the constant in the logarithm of income process, default 0.00373
	T: the time horizon, default 2000
	S: Impulse date, default 2
	sigma1: permanent shock, default 0.14364
	sigma2: transitory shock, default 0.20615
   
    Output
    =======
    C1: the impulse response path of log consumption-income ratio regarding 
        the permanent shock
    C2: the impulse response path of og consumption-income ratio regarding 
        the transitory shock
    """
    w = np.zeros(T+S)
    X1 = np.zeros(T+S)
    X2 = np.zeros(T+S)
    C1 = np.zeros(T+S)
    C2 = np.zeros(T+S)
    w[S] = 1
    
    for t in range(1, T+S-1):
        X1[t+1] = 0.704 * X1[t] + sigma1 * w[t+1]
        X2[t+1] = X2[t] - 0.154 * X2[t-1] + sigma2 * w[t+1]
        # Here I solved the explicit equations of C on X
        C1[t+1] = C1[t] - np.exp(rho - nu) * M[0] * X1[t] + M[0] * X1[t+1]
        C2[t+1] = C2[t] - np.exp(rho - nu) * (M[1] * X2[t] + M[2] * X2[t-1]) + (M[1] * X2[t+1] + M[2] * X2[t])
       
    return C1, C2