import numpy as np
import os
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy.solvers.solveset import linsolve
from sympy import Symbol, exp, log, symbols, linear_eq_to_matrix
import scipy.linalg as la
from pprint import pprint

# Set print options for numpy
np.set_printoptions(suppress=True, threshold=3000, precision = 4)

# Define parameters
S = 2           # Impulse date
sigm1 = 0.108*1.33 *.01 # Permanent shock
sigm2 = 0.155*1.33 *.01 # Transitory shock
c = 0           # steady state consumption
k = 0           # steady state capital to income
rho = 0.00663     # rate of return on assets
nu = 0.00373     # constant in the log income process
delt = rho - nu       # discount rate
nt = 10         # numeric tolerence

# Define shocks
# Z0_1 = np.zeros((5,1))
# Z0_1[2,0] = sigm1
# Z0_1[1,0] = -sigm1
# Z0_1[0,0] = -sigm1*k
# Z0_2 = np.zeros((5,1))
# Z0_2[3,0] = sigm2
# Z0_2[1,0] = -sigm2
# Z0_2[0,0] = -sigm2*k

# Define Matrix Sy, B, Bx
## Here we use Su, Sy, Sv, Fy as row vectors for convenience. The counterparts
## in the note are (Su)', (Sy)', (Sv)', (Fy)'
# Sy = np.array([0.704, 0, -0.154])
# B = np.hstack([Z0_1, Z0_2])
#
# Bx = B[2:,:]
Z0_1 = np.zeros((5,1))
Z0_1[2,0] = sigm1
Z0_2 = np.zeros((5,1))
Z0_2[3,0] = sigm2


# Define Matrix Sy, B, Bx
## Here we use Su, Sy, Sv, Fy as row vectors for convenience. The counterparts
## in the note are (Su)', (Sy)', (Sv)', (Fy)'
Sy = np.array([0.704, 0, -0.154])
B = np.hstack([Z0_1, Z0_2])
Bx = B[2:,:]
# print(Z0_2)
# Define Fy
Fy = np.array([sigm1,sigm2])



#==============================================================================
# Function: Solve for J matrix and matrix for stable dynamics
#==============================================================================
def solve_habit_persistence(alpha=0.5, psi=0.3, eta=2, print_option=False):
    """
    This function solves the matrix J and stable dynamic matrix A
    in Habit Persistence Section of the RA notes. Here we assume
    I is a 7X7 identity matrx.

    Input
    ==========
    alpha: share parameter on habit. Default 0.5
    eta: elasticity of substitution. Default 2
    psi: depreciation rate. 0 <= psi < 1. Default 0.3
    print_option: boolean, True if print the detailed results. Default False

    Output
    ==========
    J: 7X7 matrix
    A: 5X5 stable dynamic matrix
    N1, N2: stable dynamics for costates

    """
    ##== Parameters and Steady State Values ==##
    # Parameters
    eta = eta;
    psi = psi;
    alph = alpha;

    # h
    h = Symbol('h')
    Eq = exp(h)*exp(nu) - (exp(-psi)*exp(h) + (1 - exp(-psi))*exp(c))
    h = solve(Eq,h)[0]

    # u
    u = 1/(1 - eta)*log((1 - alph)*exp((1 - eta)*c) + alph*exp((1 - eta)*h))

    # mh
    exp_mh = Symbol('exp_mh')
    Eq = exp(-delt - psi - nu)*exp_mh - (exp_mh - alph*exp(-delt - nu)*exp((eta - 1)*u - eta*h))
    # print(eta, psi, alph, Eq)
    try:
        exp_mh = solve(Eq,exp_mh)[0]
    except:
        print("Problematic")
        print(eta, psi, alph)
        print(Eq)
        exp_mh = 0

    # mk
    mk = Symbol('mk')
    Eq = (1 - alph)*exp((eta - 1)*u - eta*c) - \
         (exp(mk) - (1 - exp(-psi))*exp_mh)
    mk = solve(Eq,mk)[0]

    # Print the values
    if print_option:
        print('==== 1. Calculate parameters and Steady State Values ====')
        print('u =', u)
        print('exp_mh =', exp_mh)
        print('\n')


    ##== Construct Ut ==##
    if print_option:
        print('==== 2. Solve Ut in terms of Z ====')
    # Z^{1}_{t}
    MKt, MHt, Kt, Ct, Ht, X1t, X2t, X2tL1 = symbols('MKt MHt Kt Ct Ht X1t X2t X2tL1')
    # Z^{1}_{t+1}
    MKt1, MHt1, Kt1, Ct1, Ht1, X1t1, X2t1 = symbols('MKt1 MHt1 Kt1 Ct1 Ht1 X1t1 X2t1')
    Ut, Ut1 = symbols('Ut Ut1')

    # Equation (3, )
    Ut = (1 - alph)*exp((eta - 1)*(u - c))*Ct + alph*exp((eta - 1)*(u - h))*Ht
    # New for equations
    Ut1 = (1 - alph)*exp((eta - 1)*(u - c))*Ct1 + alph*exp((eta - 1)*(u - h))*Ht1
    # Print Ut
    if print_option:
        print('Ut =', Ut)
        print('\n')


    ##== Solve for Linear system L and J  ==##
    if print_option:
         print('==== 3. Solve matrix L and J ====')

    # Equation (25)
    Eq1 = exp(-delt + rho - nu)*(MKt1 - (0.704*X1t - 0.154*X2tL1)) - MKt

    # Equation (23)
    Eq2 = exp(-delt - psi - nu) * exp_mh *(MHt1 - (0.704*X1t - 0.154*X2tL1)) + \
          alph*exp((eta - 1)*u - eta*h - nu - delt) * ((eta - 1)*Ut1 - eta * Ht1 - \
                                            (0.704*X1t - 0.154*X2tL1)) -\
          exp_mh * MHt

    # Equation (13)
    Eq3 = Kt1 - (exp(rho - nu)*Kt - exp(-nu)*Ct - k*(0.704*X1t - 0.154*X2tL1))

    # Equation (22)
    Eq4 = Ht1 - (exp(-psi - nu)*Ht + (1 - exp(-psi - nu))*Ct - (0.704*X1t - 0.154*X2tL1))

    # Equation for X
    Eq5 = X1t1 - 0.704*X1t
    Eq6 = X2t1 - (X2t - 0.154*X2tL1)

    # Equation (24)
    Eq8 = (1 - alph)*exp((eta - 1)*u - eta*c)*((eta - 1)*Ut - eta*Ct) - \
          (exp(mk)*MKt - (1 - exp(-psi))*exp_mh*MHt)

    # Create a list of the variables in Zt1 and Zt, excluding X2t from Zt1
    # to avoid duplicates
    lead_vars = [MKt1, MHt1, Ct1, Kt1, Ht1, X1t1, X2t1]
    lag_vars = [MKt, MHt, Ct, Kt, Ht, X1t, X2t, X2tL1]
    eqs = [Eq1, Eq2, Eq8, Eq3, Eq4, Eq5, Eq6]

    try:
        L = np.array([[eq.diff(var) for var in lead_vars] for eq in eqs]).astype(np.float)
    except:
        print([[eq.diff(var) for var in lead_vars] for eq in eqs])
    J = -np.array([[eq.diff(var) for var in lag_vars] for eq in eqs]).astype(np.float)

    # Add an row to J and L to specify the X2t relationship
    L = np.hstack([L, np.zeros((len(L),1))])
    L = np.vstack([L, np.zeros((1,8))])
    L[-1,-1] = 1
    J = np.vstack([J, np.zeros((1,8))])
    J[-1,-2] = 1

    # Define a sorting criterion for the Generalized Schur decomposition which
    # pushes all the explosive eigenvalues to the lower right corner and is
    # robust to cases where one of the matrices has a zero on the diagonal
    def sort_req(alpha, beta):
        return np.abs(alpha) <= np.abs(beta)

    # Perform the Generalized Schur decomposition
    LL, JJ, a, b, Q, Z = la.ordqz(J, L, sort=sort_req)
    # print(LL)
    # print(JJ)

    # Make the last 3 entries of Z.T@Y equal zero
    G11 = Z.T[-3:,:3]
    G12 = Z.T[-3:,3:]
    N = -la.inv(G11) @ G12

    # Back out A using the solution for N (differs from notes since L isn't invertible)
    N_block = np.block([[N],[np.eye(5)]])
    L_tilde = (L@N_block)[3:]
    J_tilde = (J@N_block)[3:]
    A = la.inv(L_tilde) @ J_tilde

    return J, A, N, Ut




#==============================================================================
# Function: Output time path for log consumption responses
#==============================================================================
def habit_persistence_consumption_path(A, N, T=100, print_option=False):
    """
    This function outputs the time path of C and Z responses given the
    intial shock vector.

    Input
    ==========
    A: 5X5 stable dynamic matrix
    N: stable dynamics for costates
    T: Time periods
    print_option: boolean, True if print the detailed results. Default False

    Output
    ==========
    C1Y1_path: the path of consumption response regarding the permanent shock
    C2Y2_path: the path of consumption response regarding the transitory shock

    """
    ##== Compute path for Z ==##
    if print_option:
         print('==== 7. Add the shock to the equations ====')

    # The vector Z includes Kt, Ht, X1t, X2t, X2tL1 in that order
    # The vector E includes the endogenous vars MKt, MHt, and Ct in that order
    CY_path_list = []
    Y_path_list = []
    Z_path_list = []
    E_path_list = []

    for n, Z0 in enumerate([Z0_1, Z0_2]):
        Z_path = np.zeros((len(Z0), T))
        E_path = np.zeros((len(N), T))
        Z_path[:,0] = Z0.flatten() * 100
        E_path[:,0] = N @ Z_path[:,0]

        for t in range(1, T):
            Z_path[:,t] = A @ Z_path[:,t-1]
            E_path[:,t] = N @ Z_path[:,t]

        if n==0:
            X_path = Z_path[2]
            Y_path = np.cumsum(X_path)
        elif n==1:
            X_path = Z_path[3]
            Y_path = X_path

#         plt.plot(E_path[0] - Y_path, label=r'$MK_t - Y_t$')
        # plt.plot(E_path[1] - Y_path, label=r'$MH_t - Y_t$')
        # plt.plot(E_path[0], label=r'$MK_t$')
        # plt.plot(E_path[1], label=r'$MH_t$')
        # plt.plot(E_path[2], label=r'$C_t$')
        # plt.plot(E_path[2] + Y_path, label=r'$C_t + Y_t$')
        # plt.plot(E_path[2] + Y_path, label=r'Log Consumption')
        # plt.plot(Z_path[0], label='Kt')
        # plt.plot(Z_path[1], label='$H_t$')
        # plt.plot(Z_path[1,1:] + Y_path[1:], label=r'$H_{t+1} + Y_{t+1}$')
        # plt.plot(Z_path[1] + Y_path, label=r'$H_t + Y_t$')
        # plt.plot(Y_path, label=r'$Y_t$')
        # plt.title(r"shock = {}".format(n+1))
        # plt.xlabel("t")
        # plt.legend()
#         plt.show()

        CY_path = E_path[2] + Y_path
        CY_path_list.append(CY_path)
        Y_path_list.append(Y_path)
        Z_path_list.append(Z_path)
        E_path_list.append(E_path)
    ##== Return results ==##
    return CY_path_list, Y_path_list, Z_path_list, E_path_list


#==============================================================================
# Function: Solve for Sv
#==============================================================================
def get_Sv(J, A, N, Ut):
    """
    Solve for Sv

    Input
    =========
    (The inputs are obtained from solve_habit_persistence)
    J: matrix J
    A: stable dynamic matrix A
    N1, N2: stable dynamics for costates
    Ut: the utility function

    Output
    =========
    Sv: The row vector (Sv)'

    """
    ##== Calculate Su ==##
    Ht, Ct = symbols('Ht Ct')
    c_Ht = float(Ut.coeff(Ht))
    c_Ct = float(Ut.coeff(Ct))

    Su = c_Ct * N[2]
    Su[1] += c_Ht


    ##== Calculate Sv ==##
    """
    We rearranged the equation from the note to get: (Sv)' * A_Sv = b_Sv
    """
    b_Sv = np.array((1 - np.exp(-delt)) * Su + np.exp(-delt) * np.hstack([0,0,Sy]))
    A_Sv = (np.identity(5) - np.exp(-delt) * A)

    Sv = b_Sv @ la.inv(A_Sv)
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
    sv = xi/2 * (np.linalg.norm(np.matmul(Sv, B) + np.matmul(Sy, Bx)))**2 / (np.exp(-delt) - 1)

    return sv



#==============================================================================
# Function: Calculate Sv'B + Fy
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




#==============================================================================
# Function: Solve the habit persistence consumption and uncertainty price
#==============================================================================
def habit_consumption_and_uncertainty_price(alpha=0.5, psi=0.3, eta=2, T=100):
    """
    Create the habit persistence consumption response paths.

    Input
    ==========
    alpha: share parameter
    eta: elasticity of substitution
    psi: depreciation rate, 0≤exp⁡(−psi)<1
    T: Time periods

    Output
    ==========
    C1Y1: the path of consumption response regarding the permanent shock
    C2Y2: the path of consumption response regarding the transitory shock
    SvBFy: uncertainty price vector

    """
    # Solve the habit persistence model
    J, A, N, Ut = solve_habit_persistence(alpha = alpha, psi = psi, eta = eta, print_option=True)

    # Compute the time paths for the consumption responses
    CYs, Ys = habit_persistence_consumption_path(A, N, T=T)
    C1Y1, C2Y2 = CYs
    Y1, Y2 = Ys
    # Compute uncertainty price
    Sv = get_Sv(J, A, N, Ut)
    SvBFy = get_SvBFy(Sv) * 100
    SvBFy = [float('%.3g' % x) for x in SvBFy]

    return C1Y1, C2Y2, Y1, Y2, SvBFy

if __name__ == "__main__":
    C1Y1, C2Y2, _, _, Price = habit_consumption_and_uncertainty_price(alpha=0.00001, psi=.05, eta=35, T=81)
    plt.plot(C2Y2)
    plt.show()
