import numpy as np
np.set_printoptions(precision=4, suppress=True)
from sympy import log, exp, symbols
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.linalg as la
import matplotlib.pyplot as plt
import scipy.io as sio
import sys
import pprint
from scipy import linalg as la
import os

alpha_c = 0.00484
IoverK = 0.074
CoverI = 2.556
alpha_d = 0

def solve_model(rho, gam, T, phi1, phi2, verbose = False, transform = False,
                empirical = 1, calibrated = True, risk_free_adj = 1, shock = 1,
                verbose_ss = False):

    #######################################################
    #                Section 1: Calibration               #
    #######################################################

    if empirical == 0 or empirical == 0.5:
        beta2 = 0.0022
        zeta = 0.014
        # Calibrated params old model
        # a_k = 0.03191501061172916
        # phi = 13.353240248981844
        # A = 0.26314399999999993
        # delta = 0.007 * risk_free_adj - alpha_c * rho

        # Original, randomly selected params
        # a_k = 0.017
        # phi = 13.807 / 2
        # A = 0.052
        # delta = 0.025


        if empirical == 0:
            # Eberly Wang annual params
            # a_k = .1
            # phi2 = 100
            # phi1 = .05
            # # A = 0.1 + .004
            # A = 0.1 + .042
            # delta = .02

            # Eberly Wang quarterly params
            a_k = .1 / 4
            # phi2 = 100 * 4
            # phi1 = .05 / 4
            A = (.1 + .042) / 4
            delta = .02 / 4

        if empirical == 0.5:
            # Low adjustment cost model annual params
            # a_k = .05
            # phi2 = 3.
            # phi1 = 1. / phi1
            # A = .14
            # delta = .05

            # Low adjustment cost model quarterly params
            a_k = .05 / 4
            # phi2 = 3. * 4
            # phi1 = 1. / phi1
            A = .14 / 4
            delta = .05 / 4

        def f(c):
            Phi = (1 + phi2 * (A - np.exp(c)))**(phi1)
            Phiprime = phi2 * phi1 * (1 + phi2 * (A - np.exp(c)))**(phi1 - 1)
            k = np.log(Phi) - a_k

            if rho == 1:
                v = c + k * np.exp(-delta) / (1 - np.exp(-delta))
            else:
                v = np.log((1 - np.exp(-delta)) * np.exp(c * (1 - rho)) / (1 - np.exp(-delta + k * (1 - rho)))) / (1 - rho)

            r1 = Phiprime - (np.exp(delta) - 1) * Phi * np.exp(c * -rho + (v + k) * (rho - 1))
            return r1


        sol = opt.bisect(f, -10, np.log(A), disp = True)
        cstar = sol
        # print(f(cstar))
        def v(c):
            Phi = (1 + phi2 * (A - np.exp(c)))**(phi1)
            # Phiprime = phi2 * phi1 * (1 + phi2 * (A - np.exp(c)))**(phi1 - 1)
            k = np.log(Phi) - a_k

            if rho == 1:
                v = c + k * np.exp(-delta) / (1 - np.exp(-delta))
            else:
                v = np.log((1 - np.exp(-delta)) * np.exp(c * (1 - rho)) / (1 - np.exp(-delta + k * (1 - rho)))) / (1 - rho)

            # r1 = Phiprime - (np.exp(delta) - 1) * Phi * np.exp(c * -rho + (v + k) * (rho - 1))
            return v

        def denom(c):
            Phi = (1 + phi2 * (A - np.exp(c)))**(phi1)
            # Phiprime = phi2 * phi1 * (1 + phi2 * (A - np.exp(c)))**(phi1 - 1)
            k = np.log(Phi) - a_k
            return (1 - np.exp(-delta + k * (1 - rho)))
        def k(c):
            Phi = (1 + phi2 * (A - np.exp(c)))**(phi1)
            # Phiprime = phi2 * phi1 * (1 + phi2 * (A - np.exp(c)))**(phi1 - 1)
            k = np.log(Phi) - a_k
            return k

        dom = np.linspace(-10, np.log(A), 500)
        # plt.plot(dom, f(dom))
        # plt.plot(dom, v(dom))
        # plt.subplot(1,2,1)
        # plt.plot(dom, denom(dom), label='denominator')
        # plt.plot(dom, k(dom), label='k')
        # plt.plot(dom, np.zeros_like(dom))
        # plt.plot(dom, np.ones_like(dom) * delta / (1 - rho), label = r'$\frac{\delta}{(1 - \rho)}$')
        # plt.legend()
        # plt.subplot(1,2,2)
        # plt.plot(dom, f(dom), label = 'Root function')
        # plt.legend()
        # plt.show()

        if np.min(np.abs(cstar - np.array([-10, np.log(A)]))) < 1e-8:
            raise ValueError("Not actually solved")

        Phi = (1 + phi2 * (A - np.exp(cstar)))**(phi1)
        Phiprime = phi2 * phi1 * (1 + phi2 * (A - np.exp(cstar)))**(phi1 - 1)

        kstar = np.log(Phi) - a_k
        istar = np.log(A - np.exp(cstar))

        if rho == 1:
            vstar = cstar + kstar * np.exp(-delta) / (1 - np.exp(-delta))
        else:
            vstar = np.log((1 - np.exp(-delta)) * np.exp(cstar * (1 - rho)) / (1 - np.exp(-delta + kstar * (1 - rho)))) / (1 - rho)

        istar = np.log(A - np.exp(cstar))

        zstar = 0
        dstar = 0

    # Calculate parameters using empirical targets
    elif empirical == 1:
        # Use all empirical targets with all parameters free
        istar = np.log(IoverK)
        cstar = np.log(CoverI * np.exp(istar))
        delta0 = 0.007 * risk_free_adj - alpha_c * rho
        A0 = np.exp(istar) + np.exp(cstar)
        kstar = alpha_c

        def f(v0):
            if rho != 1:
                r3 = np.exp(v0) ** (1 - rho) - (1 - np.exp(-delta0)) \
                     * np.exp(cstar) ** (1 - rho) - np.exp(-delta0) \
                     * (np.exp(v0) * np.exp(kstar)) ** (1 - rho)
            else:
                r3 = np.exp(v0) ** (1 - np.exp(-delta0)) \
                     - np.exp(cstar) ** (1 - np.exp(-delta0)) \
                     * np.exp(kstar) ** (np.exp(-delta0))

            return r3

        vstar = opt.root(f, -3.2).x[0]

        def g(phi):
            Phi = (1. + phi * np.exp(istar)) ** (1. / phi)
            PhiPrime = (1. + phi * np.exp(istar)) ** (1. / phi - 1)
            return np.exp(-rho * cstar + (rho - 1) * (vstar + kstar)) \
                * (np.exp(delta0) - 1) * (Phi) \
                - PhiPrime

        phi0 = opt.root(g, 700).x[0]

        a_k0 = np.log((1. + phi0 * np.exp(istar)) ** (1. / phi0)) - kstar

        zstar = 0

        A = A0
        delta = delta0
        a_k = a_k0
        phi = phi0

    elif empirical == 2:
        raise ValueError("The specifications for C and V are not yet developed for this empirical case.")
        # Fix phi = 0 and free C/I
        I = IoverK
        istar = np.log(I)
        phi = 0
        delta = 0.007 * risk_free_adj - alpha_c * rho
        kstar = alpha_c
        G = np.exp(kstar)
        a_k = np.log(1 + I) - kstar

        if (-delta + (1 - rho) * kstar) >= 0:
            raise ValueError(("The constraint to solve for V is not"
                              "satisfied for rho = {}").format(rho))

        if rho != 1:
            C = (np.exp(delta) - 1) * (G ** (rho - 1) - np.exp(-delta)) \
                * (1 + I) / (1 - np.exp(-delta))
            V = ((1 - np.exp(-delta)) * C ** (1 - rho) \
                / (1 - np.exp(-delta) * G ** (1 - rho))) ** (1 / (1 - rho))
        else:
            C = (np.exp(delta) - 1) * G ** ((rho - 1) / (1 - np.exp(-delta))) \
                * (1 + I)
            V = C * G ** (np.exp(-delta) / (1 - np.exp(-delta)))

        cstar = np.log(C)
        vstar = np.log(V)

        A = np.exp(cstar) + np.exp(istar)

        zstar = 0

    else:
        raise ValueError("'Empirical' must be 1 or 2.")

    #######################################################
    #               Section 2: Model Solution             #
    #######################################################

    #######################################################
    #       Section 2.1: Symbolic Model Declaration       #
    #######################################################

    # Declare necessary symbols in sympy notation
    # Note that the p represents time, so k = k_t and kp = k_{t+1}

    k, kp, c, cp, v, vp, zo, zop, zt, ztp = symbols("k kp c cp v vp zo zop zt ztp")

    # k = log(K_t / K_{t-1})
    # c = log(C_t / K_t)
    # v = log(V_t / K_t)
    # zo = Z_{t,1}
    # zt = Z_{t,2}
    # d = log D_t

    # Set up the equations from the model in sympy
    # The equations come from subtracting the right side from the left of the
    # log linearized governing equations

    I = A - exp(c) # I_t / K_t
    # NEW FUNCTION
#     i = 1 + I - phi / 2 * I ** 2 # quadratic version of I; also I^*/K_t
    i = (1. + phi2 * I) ** (phi1)
    # i = 1 + log(phi * I + 1)/phi
#     phip = 1 - phi * I
    phip = phi2 * phi1 * (1. + phi2 * I) ** (phi1 - 1)
    # phip = 1 / (phi * I + 1)
    r = vp + kp + zt

    # Equation 1: Capital Evolution
    eq1 = kp - log(i) + a_k - zo

    # Equation 2: First Order Conditions on Consumption
    eq2 = log(exp(delta) - 1) - rho * c + (rho - 1) * r + \
                    log(i) - log(phip)


    # Equation 3: Value Function Evolution: rho == 1 is separate case
    if rho != 1:
        eq3 = exp(v * (1 - rho))  - ((1 - exp(-delta)) * exp((1 - rho) * c) \
                            + exp(-delta) * exp((1 - rho) * r))
    else:
        eq3 = v - (1 - exp(-delta)) * c - exp(-delta) * r

    # Equations 4 and 5: Shock Processes Evolution
    eq4 = zop - exp(-zeta) * zo
    eq5 = ztp - exp(-beta2) * zt



    eqs = [eq1, eq2, eq3, eq4, eq5]
    lead_vars = [kp, cp, vp, zop, ztp]
    current_vars = [k, c, v, zo, zt]

    substitutions = {k:kstar, kp:kstar, c:cstar, cp:cstar,
                     v:vstar, vp:vstar,
                     zo:zstar, zop:zstar, zt:zstar, ztp:zstar}
    # print(substitutions)

    #######################################################
    #Section 2.2: Generalized Schur Decomposition Solution#
    #######################################################

    # Take the appropriate derivatives and evaluate at steady state
    Amat = np.array([[eq.diff(var).evalf(subs=substitutions) for \
                      var in lead_vars] for eq in eqs]).astype(np.float)
    B = -np.array([[eq.diff(var).evalf(subs=substitutions) for var in \
                       current_vars] for eq in eqs]).astype(np.float)

    # print(Amat)
    # print(B)
    # print(la.inv(Amat[:3, :3]))

    # Substitute for k and c to reduce A and B to 2x2 matrices, noting that:
    # A[0,0]kp - B[0,1]c = zo
    # A[1,0]kp - B[1,1]c = B[1,4]zt - A[1,2]vp

    M = np.array([[Amat[0,0], -B[0,1]],[Amat[1,0], -B[1,1]]])
    Minv = la.inv(M)

    # kp = Minv[0,0] * zo + Minv[0,1] * (B[1,4]zt - A[1,2]vp)      (1)
    # c  = Minv[1,0] * zo + Minv[1,1] * (B[1,4]zt - A[1,2]vp)      (2)

    # So the system can be reduced in the following way:
    Anew = np.copy(Amat[2:,2:])
    Bnew = np.copy(B[2:,2:])

    # Update the column of Anew corresponding to vp, subbing in with (1)
    Anew[:,0] += Minv[0,1] * Amat[2:,0] * (-Amat[1,2])
    # Update the column of Bnew corresponding to zo, subbing in with (1)
    Bnew[:,1] -= Minv[0,0] * Amat[2:,0]
    # Update the column of Bnew corresponding to zt, subbing in with (1)
    Bnew[:,2] -= Minv[0,1] * Amat[2:,0] * B[1,4]

    # Update the column of Anew corresponding to vp, subbing in with (2)
    Anew[:,0] -= Minv[1,1] * B[2:,1] * (-Amat[1,2])
    # Update the column of Bnew corresponding to zo, subbing in with (2)
    Bnew[:,1] += Minv[1,0] * B[2:,1]
    # Update the column of Bnew corresponding to zt, subbing in with (2)
    Bnew[:,2] += Minv[1,1] * B[2:,1] * B[1,4]

    # Compute the generalized Schur decomposition of the reduced A and B,
    # sorting so that the explosive eigenvalues are in the bottom right

    BB, AA, a, b, Q, Z = la.ordqz(Bnew, Anew, sort='iuc')

    total_dim = len(Anew)
    # a/b is a vector of the generalized eiganvals
    exp_dim = len(a[np.abs(a/b) > 1])
    stable_dim = total_dim - exp_dim

    if verbose:
        print("Rho = {}".format(rho))
        # print(-delta + (1 - rho) * kstar)
        print(("{} out of {} eigenvalues were found to be"
               " unstable.").format(exp_dim, total_dim))

    J1 = Z.T[stable_dim:,:exp_dim][0][0]
    J2 = Z.T[stable_dim:,exp_dim:][0]

    # J1v = J2 @ [zo, zt]
    v_loading = -(J2/J1)

    # Recall the following identities:
    # kp = Minv[0,0] * zo + Minv[0,1] * (B[1,4]zt - A[1,2]vp)      (1)
    # c  = Minv[1,0] * zo + Minv[1,1] * (B[1,4]zt - A[1,2]vp)      (2)

    # Rewrite as
    # kp = -Minv[0,1]*A[1,2]vp + Minv[0,0]zo + Minv[0,1]*B[1,4]zt  (1)
    # c  = -Minv[1,1]*A[1,2]vp + Minv[1,0]zo + Minv[1,1]*B[1,4]zt  (2)

    k_loading = - Minv[0,1] * Amat[1,2] * v_loading
    c_loading = - Minv[1,1] * Amat[1,2] * v_loading * np.exp(-zeta)

    # Add the zo and zt specific dependencies to each entry of each vector
    k_loading += np.array([Minv[0,0], Minv[0,1] * B[1,4]]) * np.exp(zeta)
    c_loading += np.array([Minv[1,0], Minv[1,1]])

    istar = np.log(A - np.exp(cstar))
    i_loading = (-exp(cstar) * c_loading / exp(istar)).astype(np.float)

    slopes1 = [k_loading[0], c_loading[0], i_loading[0], v_loading[0]]
    slopes2 = [k_loading[1], c_loading[1], i_loading[1], v_loading[1]]

    #######################################################
    #          Section 3: First Order Adjustments         #
    #######################################################

    # sigz = np.array([.00011, .00025, 0.0])
    # scriptB = np.array([[0.012, 0.027, 0, 0],[0, 0, 0.132, 0]]) * 0.01
    # sigk = np.array([.00477, 0.0, 0.0, 0.0])
    # scriptK = np.array([0.481, 0, 0, 0]) * 0.01

    # Create first order adjustments on constant terms
    # if verbose:
    #     print("Making first order adjustments to constants.")

    if transform:
        sig = np.vstack((sigk, sigz))
        s1 = sigk + sigz / (1 - np.exp(-zeta))
        s1 = s1 / la.norm(s1)
        s2 = s1[::-1] * np.array([-1,1])
        s = np.column_stack((s2, s1))
        snew = sig @ s

        sigk = snew[0][::-1] * np.array([1,-1])
        sigz = snew[1][::-1] * np.array([1,-1])

        #print(np.array([sigk,sigz]) * 100)

    selector = np.zeros(4)
    selector[shock - 1] = 1
    B = np.array([[0.011, 0.025, 0, 0],
                  [0, 0, 0.119, 0]])
    sigk = np.array([0.477, 0, 0, 0])
    # sigk = np.array([0.477, 0, 0, 0]) * 0.01
    sigd = np.array([0, 0, 0, 0])

    # adjustment = - (1 - gam) / 2 * la.norm(v_loading * sigz + sigk) ** 2
    # adjustment = - (1 - gam) / 2 * la.norm(v_loading @ scriptB + scriptK) ** 2
    # adjustments = la.solve((Amat - B)[:3,:3], np.array([0,0,adjustment]))

    # print(adjustments)
    # kstar += adjustments[0]
    # cstar += adjustments[1]
    # vstar += adjustments[2]
    # istar = log(A - np.exp(cstar))

    levels = [kstar, cstar, istar, vstar]

    if verbose:
        print("Log Levels: k, c, i, v")
        print(levels)
        print("Log slopes, growth shock: k, c, i, v")
        print(slopes1)
        print("Log slopes, preference shock: k, c, i, v")
        print(slopes2)
        print("\n")

    if verbose_ss:
        print("Rho = {}:".format(rho))
        print("\tLog capital growth: \t\t{0:.3f}".format(levels[0], 3))
        print("\tConsumption to capital: \t{0:.3f}".format(np.exp(levels[1]), 3))
        print("\tInvestment to capital: \t\t{0:.3f}".format(np.exp(levels[2]), 3))
        print("\tQ: \t\t\t\t{0:.3f}".format( (1 - np.exp(-delta)) * np.exp(levels[3] * (rho - 1)) * np.exp(levels[1] * -rho), 3) )

    Atransition = np.array([[np.exp(-zeta), 0], [0, np.exp(-beta2)]])


    #######################################################
    #       Section 4: Impulse Response Generation        #
    #######################################################

    Z = np.zeros((2,T))
    Z[:,0] = B@selector
    for i in range(1,T):
        Z[:,i] = Atransition @ Z[:,i-1]

    if shock == 1 or shock == 2:
        pass
        # bool_marker = np.array([1,0])
    elif shock == 3:
        pass
        # bool_marker = np.array([0, 1])
    else:
        raise ValueError("'shock' parameter must be set to 1, 2, or 3.")

    K = np.zeros(T)
    S = np.zeros(T)
    C = np.zeros(T)
    I = np.zeros(T)

    K[0] = sigk @ selector
    for p in range(1,T):
        K[p] = K[p-1] + k_loading @ Z[:,p]

    S[0] = -rho * c_loading @ Z[:,0] + (1 - rho) * Z[1,0] + \
                 (rho - gam) * ((v_loading @ B) + sigk + sigd)[shock - 1] \
                  - rho * K[0]

    for p in range(1,T):
        S[p] = S[p-1] - rho * c_loading @ (Z[:,p] - Z[:,p-1]) \
                - rho * k_loading @ Z[:,p] + (1 - rho) * Z[1,p]

    C = c_loading @ Z + K

    I = i_loading @ Z + K

    return levels, slopes1, slopes2, np.array([-S.astype(np.float), K.astype(np.float),
                                     C.astype(np.float), I.astype(np.float)])


if __name__ == "__main__":
    # Can be used for simple disgnostics
    r = .5
    # r = float(sys.argv[1])
    emp = 0
    phi2 = 400
    solve_model(r, 10, 0.014, 200, 5./phi2, phi2, risk_free_adj = 1,
                                 empirical = emp,
                                 transform = False, shock = 1, verbose = True)
