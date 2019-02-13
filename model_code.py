import numpy as np
# np.set_printoptions(precision=4, suppress=True)
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

# alpha_c = 0.00373
alpha_c = 0.00484
# IoverK = IoverK
IoverK = 0.074
# CoverI = 1
CoverI = 2.556

def solve_model(rho, gam, zeta, T, verbose = False, transform = False,
                empirical = 1, calibrated = True, risk_free_adj = 1, shock = 1):

    # Calculate parameters using empirical targets
    if empirical == 1:
        # Use all empirical targets with all parameters free
        istar = np.log(IoverK)
        cstar = np.log(CoverI * np.exp(istar))
        delta0 = 0.007 * risk_free_adj - alpha_c * rho
        A0 = np.exp(istar) + np.exp(cstar)
        kstar = alpha_c

        # def f(x):
        #     return np.exp(x * (1 - rho)) - (1 - np.exp(-delta0)) * \
        #     np.exp(cstar * (1 - rho)) - \
        #     np.exp(-delta0 + (kstar + x) * (1 - rho))
        #
        # dom = np.linspace(-1, 10, 500)
        # plt.plot(dom, f(dom))
        # plt.hlines(0, -1, 10)
        # plt.show()
        # plt.clf()

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

        # fig = plt.figure()
        # dom = np.linspace(100, 10000, 10000)
        # plt.plot(dom, g(dom))
        # plt.show()

        a_k0 = np.log((1. + phi0 * np.exp(istar)) ** (1. / phi0)) - kstar
        print("Resids size",la.norm(np.array([f(vstar), g(phi0)])))
        zstar = 0
        #
        # def g(x):
        #     v0, phi, a_k = x
        #     # Phi = 1 + np.exp(istar) - phi/2 * np.exp(istar)**2
        #     # PhiPrime = 1 - phi * np.exp(istar)
        #     # Phi = 1 + np.log(1 + phi * np.exp(istar)) / phi
        #     # PhiPrime = 1 / (1 + phi * np.exp(istar))
        #     Phi = (1. + phi * np.exp(istar)) ** (1. / phi)
        #     PhiPrime = (1. + phi * np.exp(istar)) ** (1. / phi - 1)
        #
        #     if rho != 1:
        #         r3 = np.exp(v0) ** (1 - rho) - (1 - np.exp(-delta0)) \
        #              * np.exp(cstar) ** (1 - rho) - np.exp(-delta0) \
        #              * (np.exp(v0) * np.exp(kstar)) ** (1 - rho)
        #     else:
        #         r3 = np.exp(v0) ** (1 - np.exp(-delta0)) \
        #              - np.exp(cstar) ** (1 - np.exp(-delta0)) \
        #              * np.exp(kstar) ** (np.exp(-delta0))
        #
        #     resids = np.array([
        #         np.exp(kstar) - (Phi) * np.exp(-a_k),
        #         np.exp(-rho * cstar + (rho - 1) * (v0 + kstar)) \
        #             * (np.exp(delta0) - 1) * (Phi) \
        #             - PhiPrime,
        #         r3
        #     ])
        #     return resids
        #
        # vstar, phi0, a_k0 = opt.root(g, np.array([-3.2, 13.807, .03])).x
        # zstar = 0

        # print("SS residuals", g([vstar, phi0, a_k0]))

        A = A0
        delta = delta0
        a_k = a_k0
        # print("rho: ", rho)
        # print("vstar: ", vstar)
        # print("phi: ", phi0)
        # print("A: ", A0)
        # print("a_k: ", a_k)
        # print("residuals", g([vstar, phi0, a_k0]))
        if calibrated:
            phi = phi0
        else:
            phi = 10

        """
            # This means that we have changed a parameter and the steady state
            # calculations from before are no longer valid. Therefore the
            # calibrated parameters and the changed parameters are used to
            # find a new steady state

            def root_fun(C):
                I_ = A - C
                # Phi = 1 + I_ - phi/2 * I_**2
                # PhiPrime = 1 - phi * I_
                Phi = (1. + phi * I_) ** (1. / phi)
                PhiPrime = (1. + phi * I_) ** (1. / phi - 1)
                # Phi = 1 + np.log(1 + phi * I_) / phi
                # PhiPrime = 1 / (1 + phi * I_)
                K = (Phi) * np.exp(-a_k)
                if rho != 1:
                    residual = (np.exp(delta) - 1) * C ** -rho \
                        * (K ** (rho - 1) - np.exp(-delta)) * (Phi) \
                        - PhiPrime * (1 - np.exp(-delta)) \
                        * C ** (1 - rho)
                else:
                    residual = (np.exp(delta) - 1) * C ** -1 \
                        * K ** ((rho - 1) / (1 - np.exp(-delta))) * (Phi) \
                        - (1 - phi * I_)
                return residual

            dom = np.linspace(1e-5, A, 100)
            y = root_fun(dom)

            ypos = y > 0
            yneg = y < 0

            # There are, for some parameter sets, two roots to this
            # equation on the interval from 0 to A. We want to identify
            # these situations and use the root which correstponds to V > 0
            if (ypos[0] and ypos[-1] and (False in ypos)):
                area = dom[yneg]
                C1 = opt.brentq(root_fun, dom[0], area[0])
                C2 = opt.brentq(root_fun, area[-1], dom[-1])

                I1 = A - C1
                # K1 = (1 + I1 - phi/2 * I1**2) * np.exp(-a_k)
                K1 = (1. + phi * I1) ** (1. / phi) * np.exp(-a_k)
                # K1 = (1 + np.log(1 + phi * I1) / phi) * np.exp(-a_k)
                if rho != 1:
                    V1 = (1 - np.exp(-delta)) * C1 ** (1 - rho) \
                         / (1 - np.exp(-delta) *  K1 ** (1 - rho))
                else:
                    V1 = C1 ** (1 - np.exp(-delta)) * K1 ** (np.exp(-delta))

                I2 = A - C2
                # K2 = (1 + I2 - phi/2 * I2**2) * np.exp(-a_k)
                K2 = (1. + phi * I2) ** (1. / phi) * np.exp(-a_k)
                # K2 = (1 + np.log(1 + phi * I2) / phi) * np.exp(-a_k)
                if rho != 1:
                    V2 = (1 - np.exp(-delta)) * C2 ** (1 - rho) \
                         / (1 - np.exp(-delta) * K2 ** (1 - rho))
                else:
                    V2 = C2 ** (1 - np.exp(-delta)) * K2 ** (np.exp(-delta))

                if V1 > 0:
                    Cstar = C1
                else:
                    Cstar = C2

            elif (yneg[0] and yneg[-1] and (False in yneg)):
                area = dom[ypos]
                C1 = opt.brentq(root_fun, dom[0], area[0])
                C2 = opt.brentq(root_fun, area[-1], dom[-1])

                I1 = A - C1
                # K1 = (1 + I1 - phi/2 * I1**2) * np.exp(-a_k)
                K1 = (1. + phi * I1) ** (1. / phi) * np.exp(-a_k)
                # K1 = (1 + np.log(1 + phi * I1) / phi) * np.exp(-a_k)
                V1 = (1 - np.exp(-delta)) * C1 ** (1 - rho) \
                     / (1 - np.exp(-delta) * K1 ** (1 - rho))
                I2 = A - C2
                # K2 = (1 + I2 - phi/2 * I2**2) * np.exp(-a_k)
                K2 = (1. + phi * I2) ** (1. / phi) * np.exp(-a_k)
                # K2 = (1 + np.log(1 + phi * I2) / phi) * np.exp(-a_k)
                if rho != 1:
                    V2 = (1 - np.exp(-delta)) * C2 ** (1 - rho) \
                         / (1 - np.exp(-delta) *K2 ** (1 - rho))
                else:
                    V2 = C2 ** (1 - np.exp(-delta)) * K2 ** (np.exp(-delta))

                if V1 > 0:
                    Cstar = C1
                else:
                    Cstar = C2

            else:
                Cstar = opt.brentq(root_fun, 1e-5, A)

            Istar = A - Cstar
            # Kstar = (1 + Istar - phi/2 * Istar**2) * np.exp(-a_k)
            Kstar = (1. + phi * Istar) ** (1. / phi) * np.exp(-a_k)
            # Kstar = (1 + np.log(1 + phi * Istar) / phi) * np.exp(-a_k)
            if rho != 1:
                Vstar = ((1 - np.exp(-delta)) * Cstar ** (1 - rho)) \
                        / (1 - np.exp(-delta) * Kstar ** (1 - rho))
                        # This is actually Vstar ** (1 - rho)

                Vstar = Vstar ** (1 / (1 - rho))
            else:
                Vstar = Cstar ** (1 - np.exp(-delta)) \
                        * Kstar ** (np.exp(-delta))
                Vstar = Vstar ** (1 / (1 - np.exp(-delta)))

            cstar = np.log(Cstar)
            istar = np.log(Istar)
            kstar = np.log(Kstar)
            vstar = np.log(Vstar)
            zstar = 0
        """

    elif empirical == 2:
        raise ValueError("The specifications for C and V are not yet developed for this empirical case.")
        # Fix phi = 0 and free C/I
        I = IoverK
        istar = np.log(I)
        phi = 0
        delta = 0.008 * risk_free_adj - alpha_c * rho
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

        # print("rho: ", rho)
        # print("vstar: ", vstar)
        # print("phi: ", phi)
        # print("A: ", A)
        # print("a_k: ", a_k)
        # print("cstar", np.exp(cstar))
        # print("consumption investment", np.exp(cstar) / np.exp(istar))

    else:
        raise ValueError("'Empirical' must be 1 or 2.")

    print(A, phi, a_k)

    # Declare necessary symbols in sympy notation
    # Note that the p represents time, so k = k_t and kp = k_{t+1}
    k, kp, c, cp, v, vp, z, zp = symbols("k kp c cp v vp z zp")

    # Set up the equations from the model in sympy
    # The equations come from subtracting the right side from the left of the
    # log linearized governing equations

    I = A - exp(c) # I_t / K_t
    # i = 1 + I - phi / 2 * I ** 2 # Joe's version of I; also I^*/K_t
    i = (1. + phi * I) ** (1. / phi)
    # i = 1 + log(phi * I + 1)/phi
    # phip = 1 - phi * I
    phip = (1. + phi * I) ** (1. / phi - 1)
    # phip = 1 / (phi * I + 1)
    r = vp + kp

    # Equation 1: Capital Evolution
    eq1 = kp - log(i) + a_k - z

    # Equation 2: First Order Conditions on Consumption
    eq2 = log(exp(delta) - 1) - rho * c + (rho - 1) * r + \
                    log(i) - log(phip)

    # Equation 3: Value Function Evolution: rho == 1 is separate case
    if rho != 1:
        eq3 = exp(v * (1 - rho))  - ((1 - exp(-delta)) * exp((1 - rho) * c) \
                            + exp(-delta) * exp((1 - rho) * r))
    else:
        eq3 = v - (1 - exp(-delta)) * c - exp(-delta) * r

    # Equation 4: Shock Process Evolution
    eq4 = zp - exp(-zeta) * z

    eqs = [eq1, eq2, eq3, eq4]
    lead_vars = [kp, cp, vp, zp]
    current_vars = [k, c, v, z]

    substitutions = {k:kstar, kp:kstar, c:cstar, cp:cstar,
                     v:vstar, vp:vstar, z:zstar, zp:zstar}


    # Take the appropriate derivatives and evaluate at steady state
    Amat = np.array([[eq.diff(var).evalf(subs=substitutions) for \
                      var in lead_vars] for eq in eqs]).astype(np.float)
    B = -np.array([[eq.diff(var).evalf(subs=substitutions) for var in \
                       current_vars] for eq in eqs]).astype(np.float)

    # Substitute for k and c to reduce A and B to 2x2 matrices
    # A[0,0]k_{t+1} - B[0,1]c = z
    # A[1,0]k_{t+1} - B[1,1]c = -A[1,2]v_{t+1}
    M = np.array([[Amat[0,0], -B[0,1]],[Amat[1,0], -B[1,1]]])
    Minv = la.inv(M)
    # k_{t+1} = Minv[0,0]*z_t + Minv[0,1]*(-A[1,2]*v_{t+1}) (1)
    # c = Minv[1,0]*z_t + Minv[1,1]*(-A[1,2]*v_{t+1})       (2)

    # So the system can be reduced in the following way:
    Anew = np.copy(Amat[2:,2:])
    Bnew = np.copy(B[2:,2:])


    # Update the column of Anew corresponding to v_{t+1}, subbing in with (1)
    Anew[:,0] += Minv[0,1] * Amat[2:,0] * (-Amat[1,2])
    # Update the column of Bnew corresponding to z_t, subbing in with (1)
    Bnew[:,1] -= Minv[0,0] * Amat[2:,0]

    # Update the column of Anew corresponding to v_{t+1}, subbing in with (2)
    Anew[:,0] -= Minv[1,1] * B[2:,1] * (-Amat[1,2])
    # Update the column of Bnew corresponding to z_t, subbing in with (2)
    Bnew[:,1] += Minv[1,0] * B[2:,1]

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
    J2 = Z.T[stable_dim:,exp_dim:][0][0]

    v_loading = -J2/J1
    k_loading = Minv[0,0] * np.exp(zeta) - Minv[0,1] * Amat[1,2] * v_loading
    c_loading = Minv[1,0] - np.exp(-zeta) * Minv[1,1] * Amat[1,2] * v_loading

    istar = log(A - np.exp(cstar))
    i_loading = -exp(cstar) * c_loading / exp(istar)

    slopes = [k_loading, c_loading, i_loading, v_loading]

    sigz = np.array([.00011, .00025])
    sigk = np.array([.00477, 0.0])

    # Create first order adjustments on constant terms
    if verbose:
        print("Making first order adjustments to constants.")

    if transform:
        sig = np.vstack((sigk, sigz))
        s1 = sigk + sigz / (1 - np.exp(-zeta))
        s1 = s1 / la.norm(s1)
        s2 = s1[::-1] * np.array([-1,1])
        s = np.column_stack((s2, s1))
        snew = sig @ s

        sigk = snew[0][::-1] * np.array([1,-1])
        sigz = snew[1][::-1] * np.array([1,-1])

        print(np.array([sigk,sigz]) * 100)

    adjustment = - (1 - gam) / 2 * la.norm(v_loading * sigz + sigk) ** 2
    adjustments = la.solve((Amat - B)[:3,:3], np.array([0,0,adjustment]))

    # print(adjustments)

    kstar += adjustments[0]
    cstar += adjustments[1]
    vstar += adjustments[2]
    istar = log(A - np.exp(cstar))

    levels = [kstar, cstar, istar, vstar]

    if verbose:
        print("Log Levels: k, c, i, v")
        print(levels)
        print("Log slopes: k, c, i, v")
        print(slopes)
        print("\n")

    z1 = np.zeros(T)
    z1[0] = sigz[0]
    for i in range(1,T):
        z1[i] = np.exp(-zeta) * z1[i-1]

    z2 = np.zeros(T)
    z2[0] = sigz[1]
    for i in range(1,T):
        z2[i] = np.exp(-zeta) * z2[i-1]

    if shock == 1:
        Z = np.copy(z1)
    elif shock == 2:
        Z = np.copy(z2)
    else:
        raise ValueError("'Shock' must be 1 or 2.")

    K = np.zeros(T)
    S = np.zeros(T)
    C = np.zeros(T)
    I = np.zeros(T)

    K[0] = sigk[shock - 1]
    for p in range(1,T):
        K[p] = K[p-1] + k_loading * Z[p]

    S[0] = -rho * c_loading * Z[0] + (rho - gam) * (v_loading * Z[0] + \
                              sigk[shock - 1]) - rho * K[0]
    for p in range(1,T):
        S[p] = S[p-1] - rho * c_loading * (Z[p] - Z[p-1]) \
                - rho * k_loading * Z[p]

    C = c_loading * Z + K

    I = i_loading * Z + K

    return levels, slopes, np.array([-S.astype(np.float), K.astype(np.float),
                                     C.astype(np.float), I.astype(np.float)])


if __name__ == "__main__":
    pass
