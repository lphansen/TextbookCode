import numpy as np
import os
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol, exp, log, symbols, linear_eq_to_matrix

# Set print options for numpy
np.set_printoptions(suppress=True, threshold=3000)

# Define parameters
T = 2000        # Time horizon
S = 2           # Impulse date
σ1 = 0.108*1.33 # Permanent shock
σ2 = 0.155*1.33 # Transitory shock
c = 0           # steady state consumption
ρ = 0.00663     # rate of return on assets
ν = 0.00373     # constant in the log income process
δ = ρ - ν       # discount rate
nt = 10         # numeric tolerence

# Define shocks
Z0_1 = np.zeros((5,1))
Z0_1[2,0] = σ1
Z0_2 = np.zeros((5,1))
Z0_2[3,0] = σ2


# Define Matrix Sy, B, Bx
## Here we use Su, Sy, Sv, Fy as row vectors for convenience. The counterparts
## in the note are (Su)', (Sy)', (Sv)', (Fy)'
Sy = np.array([0.704, 0, -0.154])
B = np.hstack([Z0_1, Z0_2])
Bx = B[2:,:]

# Define Fy
Fy = np.array([σ1,σ2])



#==============================================================================
# Function: Solve for J matrix and matrix for stable dynamics
#==============================================================================
def solve_habit_persistence(alpha=0.5, eta=2, psi=0.3, print_option=False):
    """
    This function solves the matrix J and stable dynamic matrix A
    in Habit Persistence Section of the RA notes. Here we assume
    I is a 7X7 identity matrx.
    
    Input
    ==========
    alpha: share parameter on habit. Default 0.5
    eta: elasticity of substitution. Default 2
    psi: depreciation rate. 0 <= ψ < 1. Default 0.3
    print_option: boolean, True if print the detailed results. Default False
    
    Output
    ==========
    J: 7X7 matrix
    A: 5X5 stable dynamic matrix
    N1, N2: stable dynamics for costates
    
    """
    ##== Parameters and Steady State Values ==##
    # Parameters 
    η = eta;
    ψ = psi;
    α = alpha;
    
    # h
    h = Symbol('h')
    Eq = exp(h)*exp(ν) - (exp(-ψ)*exp(h) + (1 - exp(-ψ))*exp(c))
    h = solve(Eq,h)[0]
    
    # u
    u = 1/(1 - η)*log((1 - α)*exp((1 - η)*c) + α*exp((1 - η)*h))
    
    # mh
    mh = Symbol('mh')
    Eq = exp(-δ - ψ - ν)*exp(mh) - (exp(mh) - α*exp((η - 1)*u - η*h))
    mh = solve(Eq,mh)[0]
   
    # mk
    mk = Symbol('mk')
    Eq = (1 - α)*exp((η - 1)*u - η*c) - \
         (exp(-δ - ν)*exp(mk) - exp(-δ - ν)*(1 - exp(-ψ))*exp(mh))
    mk = solve(Eq,mk)[0]
    
    # Print the values
    if print_option:
        print('==== 1. Calculate parameters and Steady State Values ====')
        print('u =', u)
        print('mh =', mh)
        print('\n')  
    
    
    ##== Construct Ut and Ct ==##
    if print_option:
        print('==== 2. Solve Ct and Ut in terms of Z ====')
    # Z^{1}_{t}
    MKt, MHt, Kt, Ht, X1t, X2t, X2tL1 = symbols('MKt MHt Kt Ht X1t X2t X2tL1')
    # Z^{1}_{t+1}
    MKt1, MHt1, Kt1, Ht1, X1t1, X2t1 = symbols('MKt1 MHt1 Kt1 Ht1 X1t1 X2t1')
    Ct, Ut = symbols('Ct Ut')
    
    # Equation (3)
    Eq1 = Ut - ((1 - α)*exp((η - 1)*(u - c))*Ct + α*exp((η - 1)*(u - h))*Ht)
    
    # Equation (5)
    Eq2 = (1 - α)*exp((η - 1)*u - η*h)*((η - 1)*Ut - η*Ct) - \
          (exp(-δ - ν)*(exp(mk)*MKt1 - (1 - exp(-ψ))*exp(mh)*MHt1) + \
           exp(-δ - ν)*(exp(mk) - (1 - exp(-ψ))*exp(mh))*(0.704*X1t1 - 0.154*X2t))
        
    # Solve the system    
    sol = solve([Eq1, Eq2], Ct, Ut, set=True)[1]
    sol = list(sol)[0]
    Ct, Ut = sol
    
    # Print Ct and Ut
    if print_option:
        print('Ct =', Ct)
        print('Ut =', Ut)
        print('\n')
    
    
    ##== Solve for Linear system L and J  ==##
    if print_option:
         print('==== 3. Solve matrix J ====')
    # Equation (6)
    Eq1 = MKt1 - (exp(δ - ρ + ν)*MKt + (0.704*X1t - 0.154*X2tL1))
    
    # Equation (4)
    Eq2 = MHt1 - \
          (exp(δ + ψ + ν - mh)*(exp(mh)*MHt - \
          α*exp((η - 1)*u - η*h)*((η - 1)*Ut - η*Ht)) + \
          (0.704*X1t - 0.154*X2tL1))
    
    # Equation (1)
    Eq3 = Kt1 - (exp(ρ - ν)*Kt - exp(-ν)*Ct)
    
    # Equation (2)
    Eq4 = Ht1 - (exp(-ψ - ν)*Ht + (1 - exp(-ψ - ν))*Ct - (0.704*X1t - 0.154*X2tL1))
    
    # Equation for X
    Eq5 = X1t1 - 0.704*X1t
    Eq6 = X2t1 - (X2t - 0.154*X2tL1)
    
    # Solve the system 
    sol = solve([Eq1, Eq2, Eq3, Eq4, Eq5, Eq6], 
                Ht1, Kt1, MHt1, MKt1, X1t1, X2t1, 
                set=True)
    sol = list(sol[1])[0]
    Ht1, Kt1, MHt1, MKt1, X1t1, X2t1 = sol
    
    # Solve for J
    J,_ = linear_eq_to_matrix([MKt1,MHt1,Kt1,Ht1,X1t1,X2t1,X2t], 
                              MKt, MHt, Kt, Ht, X1t, X2t, X2tL1)
    J = np.asarray(J).astype(float)
    
    # Get Eigenvalues and Eigenvectors of J
    eigenValues, eigenVectors = np.linalg.eig(J)
    idx = np.abs(eigenValues).argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    # Print eigenvalues of J
    if print_option:
        print('The eigenvalues of J are:')
        [print(np.round(x,6)) for x in eigenValues]
        print('\n')
    
    
    ##== Initial values and matrices for stable dynamics ==##
    # Get the matrix for stable dynamics by solving the linear combination
    M = eigenVectors[:,np.abs(eigenValues)<=1+10**(-nt)] # Numeric tolerence 
    test = M[2:,:]
    sol = np.linalg.solve(test,np.identity(5))
    K = np.matmul(M,sol)
    
    # Get N1 and N2
    if print_option:
         print('==== 4. Solve N1 and N2 ====')
    N1 = K[:2,:2]
    N2 = K[:2,2:]
       
    #Get the stable dynamic matrix A
    if print_option:
         print('==== 5. Compute matrix A ====')
    JK = np.matmul(J,K)
    M_L = np.hstack([np.zeros((5,2)),np.identity(5)])
    A = np.matmul(M_L,JK)   
    
    
    ##== Check the construction results ==##
    if print_option:
         print('==== 6. Checking results ====')
         
    # First check
    check_mat = np.hstack([np.identity(2),-N1,-N2])
    check_res = np.matmul(check_mat, JK)
    check_1 = np.round(check_res, nt)==0 
    if print_option:
        print(check_1)
        
    if check_1.all():
        if print_option:
            print('First check: Passed')
    else:
        print('First check: Not passed')
        
    # Second Check
    check_eig = np.linalg.eig(A)[0]
    idx = np.abs(check_eig).argsort()[::-1]
    check_eig = check_eig[idx]
    
    chk2_list = []
    for n,x in enumerate(check_eig):
        chk2 = np.round(x, nt)==np.round(eigenValues[n+2], nt)
        if print_option:
            print(chk2)
        chk2_list.append(chk2)
    
    if all(chk2_list):
        if print_option:
            print('Passed')
    else:
        print('Second check: Not passed')

    
    
    return J, A, N1, N2, Ct, Ut




#==============================================================================
# Function: Output time path for log consumption responses
#==============================================================================
def habit_persistence_consumption_path(A, N1, N2, Ct, print_option=False):
    """
    This function outputs the time path of C and Z responses given the 
    intial shock vector. 
    
    Input
    ==========
    A: 5X5 stable dynamic matrix
    N1, N2: stable dynamics for costates
    Ct: The analytical solution of C in terms of Z
    print_option: boolean, True if print the detailed results. Default False
    
    Output
    ==========
    C1Y1_path: the path of consumption response regarding the permanent shock
    C2Y2_path: the path of consumption response regarding the transitory shock
    
    """
    ##== Compute path for Z ==##
    if print_option:
         print('==== 7. Add the shock to the equations ====')
    
    Z_path_list = []
    
    for Z0 in [Z0_1, Z0_2]: 
        Z_path = np.zeros_like(Z0)
        Z_path = np.hstack([Z_path, Z0])

        for t in range(T):
            Z = np.matmul(A,Z0)
            Z_path = np.hstack([Z_path,Z])
            Z0 = Z

        Z_path = Z_path[:,1:] 
        Z_path_list.append(Z_path)
    
    
    ##== Compute path for log consumption C + Y ==##
    if print_option:
         print('==== 8. Compute the log consumption process ====')
    
    C_path_list = []
    CY_path_list = []
    
    # Obtain coefficients of consumption-income ratio process C
    """
    Ct is the explicit formula of consumption ratio process 
    imported from function: solve_habit_persistence
    """
    MKt1, MHt1, Ht, X1t1, X2t = symbols('MKt1 MHt1 Ht X1t1 X2t')
    c_MKt1 = Ct.coeff(MKt1)
    c_MHt1 = Ct.coeff(MHt1)
    c_Ht = Ct.coeff(Ht)
    c_X1t1 = Ct.coeff(X1t1)
    c_X2t = Ct.coeff(X2t)
    
    # Compute the consumption-income ratio process C
    for n, Z_path in enumerate(Z_path_list):
        C_path = np.zeros(T)
        KH_path = Z_path[:2,:]   
        X_path = Z_path[2:,:] 
        MKMH_path = np.matmul(N1,KH_path) + np.matmul(N2,X_path)  
        MK1 = MKMH_path[0,1:T]
        MH1 = MKMH_path[1,1:T]
        H = KH_path[1,0:T-1]
        X11 = X_path[0,1:T]
        X2 = X_path[2,1:T]
        C_path[:T-1] = c_MKt1 * MK1 + c_MHt1 * MH1 + c_Ht * H + c_X1t1 * X11 + c_X2t * X2

        # Compute the income process Y
        if n==0:
            X_path = Z_path[2,:-1]
            Y_path = np.cumsum(X_path)
        elif n==1:
            X_path = Z_path[3,:-1]
            Y_path = X_path

        # Get the income process C + Y
        CY_path = C_path + Y_path
        CY_path_list.append(CY_path)

    # Extract results
    C1Y1_path = CY_path_list[0]
    C2Y2_path = CY_path_list[1] 
    
    
    ##== Return results ==##
    return C1Y1_path, C2Y2_path


#==============================================================================
# Function: Solve for Sv
#==============================================================================
def get_Sv(A, J, N1, N2, Ut):
    """
    Solve for Sv
    
    Input
    =========
    (The inputs are obtained from solve_habit_persistence)
    A: stable dynamic matrix A
    J: matrix J
    N1, N2: stable dynamics for costates
    Ut: the utility function
    
    Output
    =========
    Su: The row vector (Su)'
    
    """
    ##== Calculate Su ==##
    ## Express Ut in terms of Z_{t} and Z_{t+1}
    MKt, MHt, Kt, Ht, X1t, X2t, X2tL1 = symbols('MKt MHt Kt Ht X1t X2t X2tL1')
    MKt1, MHt1, Kt1, Ht1, X1t1, X2t1 = symbols('MKt1 MHt1 Kt1 Ht1 X1t1 X2t1')
    
    TT,_ = linear_eq_to_matrix([Ut],
                               MKt, MHt, Kt, Ht, X1t, X2t, X2tL1,
                               MKt1, MHt1, Kt1, Ht1, X1t1, X2t1)
    #t1: Ut's coefficient under Z_t
    t1 = TT[:7]
    t1 = np.array([t1]).astype(float)
    #t2: Ut's coefficient under Z_{t+1}
    t2 = TT[7:]
    t2.append(0)    #X2t is 0 in t2 entry
    t2 = np.array([t2]).astype(float)
    
    ## Get Su
    K = np.vstack([np.hstack([N1,N2]),np.identity(5)])
    JK = np.matmul(J, K)
    
    T1 = np.matmul(t1, K)
    T2 = np.matmul(t2, JK)
    
    Su = T1 + T2
    
    
    ##== Calculate Sv ==##
    """
    We rearranged the equation from the note to get: (Sv)' * A_Sv = b_Sv
    """
    b_Sv = (1 - np.exp(-δ)) * Su + np.exp(-δ) * np.hstack([0,0,Sy])
    A_Sv = (np.identity(5) - np.exp(-δ) * A)
    Sv = np.matmul(b_Sv, np.linalg.inv(A_Sv))
    
    return Sv



#==============================================================================
# Function: Calculate sv
#==============================================================================
def solve_sv(Sv, xi):
    """
    Solve sv

    Input
    ========
    Sv: the matrix Sv from get_Sv
    xi: the inverse of the risk sensitivity
    
    Output
    ========
    sv: The solution of sv
    
    """
    sv = xi/2 * (np.linalg.norm(np.matmul(Sv, B) + np.matmul(Sy, Bx)))**2 / (np.exp(-δ) - 1)
    
    return sv



#==============================================================================
# Function: Calculate SvB
#==============================================================================
def get_SvBFy(Sv):
    """
    Get the uncertainty price scaled by 1/ξ.

    Input
    ========
    Sv: the matrix Sv from get_Sv

    Output
    =========
    SvBFy: uncertainty price vector
    
    """
    SvB = np.matmul(Sv, B)
    SvBFy = SvB + Fy
    
    return SvBFy




#==============================================================================
# Function: Setup the figures
#==============================================================================
def create_fig(R, C, fs=(8,8), X=40):
    """
    Create the figure for response plots
    
    Input
    ==========
    R: Number of rows for the subplot space
    C: Number of columns for the subplot space
    fs: figure size
    
    Output
    ==========
    fig, axes: the formatted figure and axes
        
    """
    fig, axes = plt.subplots(R, C, figsize=fs)
    plt.subplots_adjust(hspace=0.5)
    
    for ax in axes:
        ax.grid(alpha=0.5)
        ax.set_xlim(0,X)
        ax.set_xlabel(r'Quarters')
    
    return fig, axes
