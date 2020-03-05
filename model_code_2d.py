import numpy as np
np.set_printoptions(precision=4, suppress=True)
from sympy import log, exp, symbols
import scipy.optimize as opt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.io as sio
import sys
import pprint
from scipy import linalg as la
import os
import pandas as pd
try:
    import plotly.graph_objs as go
    from plotly.tools import make_subplots
    from plotly.offline import init_notebook_mode, iplot
except ImportError:
    print("Installing plotly. This may take a while.")
    from pip._internal import main as pipmain
    pipmain(['install', 'plotly'])
    import plotly.graph_objs as go
    from plotly.tools import make_subplots
    from plotly.offline import init_notebook_mode, iplot

class stochastic_growth_model:
    def __init__(self, rho, phi1, phi2, a_k, A, delta, beta1, beta2):
        self.rho = rho
        self.phi1 = phi1
        self.phi2 = phi2
        self.a_k = a_k
        self.A = A
        self.delta = delta
        self.solved = False
        self.beta1 = beta1
        self.beta2 = beta2

    def find_steady_state(self):
        def f(c):
            Phi = (1 + self.phi2 * (self.A - np.exp(c)))**(self.phi1)
            Phiprime = self.phi2 * self.phi1 * \
                (1 + self.phi2 * (self.A - np.exp(c)))**(self.phi1 - 1)
            k = np.log(Phi) - self.a_k

            if self.rho == 1:
                v = c + k * np.exp(-self.delta) / (1 - np.exp(-self.delta))
            else:
                v = np.log((1 - np.exp(-self.delta)) * np.exp(c * (1 - self.rho)) / \
                           (1 - np.exp(-self.delta + k * (1 - self.rho)))) / (1 - self.rho)

            r1 = Phiprime - (np.exp(self.delta) - 1) * Phi * np.exp(c * -self.rho + (v + k) * (self.rho - 1))
            return r1

        interval_min = -40
        interval_max = np.log(self.A)

        # dom = np.linspace(interval_min, interval_max, 100)
        # plt.plot(dom, f(dom))
        # plt.show()

        self.cstar = opt.bisect(f, interval_min, interval_max, disp = True)

        if np.min(np.abs(self.cstar - np.array([interval_min, interval_max]))) < 1e-8:
            raise ValueError("Steady states not found for rho = {}. Try decreasing depreciation.".format(self.rho))

        Phi = (1 + self.phi2 * (self.A - np.exp(self.cstar)))**(self.phi1)
        Phiprime = self.phi2 * self.phi1 * \
            (1 + self.phi2 * (self.A - np.exp(self.cstar)))**(self.phi1 - 1)

        self.kstar = np.log(Phi) - self.a_k
        self.istar = np.log(self.A - np.exp(self.cstar))

        if self.rho == 1:
            self.vstar = self.cstar + self.kstar * np.exp(-self.delta) / (1 - np.exp(-self.delta))
        else:
            self.vstar = np.log((1 - np.exp(-self.delta)) * \
                           np.exp(self.cstar * (1 - self.rho)) / \
                           (1 - np.exp(-self.delta + self.kstar * (1 - self.rho)))) / \
                           (1 - self.rho)

        self.zstar = 0

    def solve_model(self):
        k, kp, c, cp, v, vp, zo, zop, zt, ztp = symbols("k kp c cp v vp zo zop zt ztp")

        # k = log(K_t / K_{t-1})
        # c = log(C_t / K_t)
        # v = log(V_t / K_t)
        # zo = Z_{t,1}
        # zt = Z_{t,2}
        # d = log(D_t / D_{t+1})

        # Note that dp = zt, so we can replace dp in the code with zt. We
        # do this for simplicity.

        # Set up the equations from the model in sympy
        # The equations come from subtracting the right side from the left of the
        # log linearized governing equations

        I = self.A - exp(c) # I_t / K_t
        i = (1. + self.phi2 * I) ** (self.phi1)
        phip = self.phi2 * self.phi1 * (1. + self.phi2 * I) ** (self.phi1 - 1)
        r = vp + kp + zt

        # Equation 1: Capital Evolution
        eq1 = kp - log(i) + self.a_k - zo

        # Equation 2: First Order Conditions on Consumption
        eq2 = log(exp(self.delta) - 1) - self.rho * c + (self.rho - 1) * r + \
                        log(i) - log(phip)

        # Equation 3: Value Function Evolution: rho == 1 is separate case
        if self.rho != 1:
            eq3 = exp(v * (1 - self.rho))  - ((1 - exp(-self.delta)) * \
                    exp((1 - self.rho) * c) + exp(-self.delta) * \
                    exp((1 - self.rho) * r))
        else:
            eq3 = v - (1 - exp(-self.delta)) * c - exp(-self.delta) * r

        # Equations 4 and 5: Shock Processes Evolution
        eq4 = zop - exp(-self.beta1) * zo
        eq5 = ztp - exp(-self.beta2) * zt

        eqs = [eq1, eq2, eq3, eq4, eq5]
        lead_vars = [kp, cp, vp, zop, ztp]
        current_vars = [k, c, v, zo, zt]

        substitutions = {k:self.kstar, kp:self.kstar, c:self.cstar,
                         cp:self.cstar, v:self.vstar, vp:self.vstar,
                         zo:self.zstar, zop:self.zstar, zt:self.zstar,
                         ztp:self.zstar}

        #######################################################
        #       Generalized Schur Decomposition Solution      #
        #######################################################

        # Take the appropriate derivatives and evaluate at steady state
        Amat = np.array([[eq.diff(var).evalf(subs=substitutions) for \
                          var in lead_vars] for eq in eqs]).astype(np.float)
        B = -np.array([[eq.diff(var).evalf(subs=substitutions) for var in \
                           current_vars] for eq in eqs]).astype(np.float)

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

        # if verbose:
        #     print("Rho = {}".format(rho))
        #     # print(-delta + (1 - rho) * kstar)
        #     print(("{} out of {} eigenvalues were found to be"
        #            " unstable.").format(exp_dim, total_dim))

        J1 = Z.T[stable_dim:,:exp_dim][0][0]
        J2 = Z.T[stable_dim:,exp_dim:][0]

        # J1v = J2 @ [zo, zt]
        self.v_loading = -(J2/J1)

        # Recall the following identities:
        # kp = Minv[0,0] * zo + Minv[0,1] * (B[1,4]zt - A[1,2]vp)      (1)
        # c  = Minv[1,0] * zo + Minv[1,1] * (B[1,4]zt - A[1,2]vp)      (2)

        # Rewrite as
        # kp = -Minv[0,1]*A[1,2]vp + Minv[0,0]zo + Minv[0,1]*B[1,4]zt  (1)
        # c  = -Minv[1,1]*A[1,2]vp + Minv[1,0]zo + Minv[1,1]*B[1,4]zt  (2)

        self.k_loading = - Minv[0,1] * Amat[1,2] * self.v_loading
        self.c_loading = - Minv[1,1] * Amat[1,2] * self.v_loading * \
            np.array([np.exp(-self.beta1), np.exp(-self.beta2)])

        # Add the zo and zt specific dependencies to each entry of each vector
        self.k_loading += np.array([Minv[0,0], Minv[0,1] * B[1,4]]) * \
            np.array([np.exp(self.beta1), np.exp(self.beta2)])
        self.c_loading += np.array([Minv[1,0], Minv[1,1]])

        self.i_loading = (-exp(self.cstar) * self.c_loading / \
                          exp(self.istar)).astype(np.float)
        self.solved = True

    def gen_impulse_response(self, shock, T, gam, B, sigk, sigd):
        selector = np.zeros(4)
        selector[int(shock) - 1] = 1

        A = np.array([[np.exp(-self.beta1), 0], [0, np.exp(-self.beta2)]])

        #######################################################
        #       Section 4: Impulse Response Generation        #
        #######################################################

        Z = np.zeros((2,T))
        Z[:,0] = B@selector
        for i in range(1,T):
            Z[:,i] = A @ Z[:,i-1]

        temp = np.zeros(T)
        temp[0] = sigd @ selector
        temp[1:] = Z[1,:-1]
        Z[1] = temp

        # Note that since dp = zt + sigma_d @ Wp, we can convert the second row
        # of Z to d by simply adding

        if shock not in [1,2,3, 4]:
            raise ValueError("'shock' parameter must be set to 1, 2, 3, or 4.")

        K = np.zeros(T)
        S = np.zeros(T)
        C = np.zeros(T)
        I = np.zeros(T)

        K[0] = sigk @ selector
        for p in range(1,T):
            K[p] = K[p-1] + self.k_loading @ Z[:,p]

        S[0] = -self.rho * self.c_loading @ Z[:,0] + (1 - self.rho) * Z[1,0] + \
                     (self.rho - gam) * ((self.v_loading @ B) + \
                         sigk + sigd)[shock - 1] - self.rho * K[0]

        for p in range(1,T):
            S[p] = S[p-1] - self.rho * self.c_loading @ (Z[:,p] - Z[:,p-1]) \
                    - self.rho * self.k_loading @ Z[:,p] + (1 - self.rho) * Z[1,p]

        C = self.c_loading @ Z + K

        I = self.i_loading @ Z + K

        self.S_response = -S.astype(np.float)
        self.K_response = K.astype(np.float)
        self.C_response = C.astype(np.float)
        self.I_response = I.astype(np.float)

    def print_model_solution_data(self):
        levels = [self.kstar, self.cstar, self.istar, self.vstar]
        slopes1 = [self.k_loading[0], self.c_loading[0], self.i_loading[0], self.v_loading[0]]
        slopes2 = [self.k_loading[1], self.c_loading[1], self.i_loading[1], self.v_loading[1]]
        if self.solved:
            print("Log Levels: k, c, i, v")
            print(levels)
            print("Log slopes, growth shock: k, c, i, v")
            print(slopes1)
            print("Log slopes, preference shock: k, c, i, v")
            print(slopes2)
            print("\n")

def bound_phi2(delta, rho, a_k, phi1phi2, a):

    def root_function(phi2):
        return phi1phi2 / phi2 * np.log(1 + phi2 * a) - a_k - delta / (1 - rho)
    sol = opt.root(root_function, 100)
    if sol.success:
        return sol
    else:
        sol = opt.root(root_function, 10)
        return sol

def plot_impulse(rhos, gammas, T, phi1, phi2, a_k, A, delta, beta1, beta2,
                 B, sigd, sigk, shock = 1, title = None):
    """
    Given a set of parameters, computes and displays the impulse responses of
    consumption, capital, the consumption-investment ratio, along with the
    shock price elacticities.

    Input
    ==========
    Note that the values of delta, phi, A, and a_k are specified within the code
    and are only used for the empirical_method = 0 or 0.5 specifications (see below).

    rhos:               The set of rho values for which to plot the IRFs.
    gamma:              The risk aversion of the model.
    betaz:              Shock persistence.
    T:                  Number of periods to plot.
    shock:              (1 or 2) Defines which of the two possible shocks to plot.
    empirical method:   Use 0 to use Eberly and Wang parameters and 0.5 for parameters
                        from a low adjustment cost setting. Further cases still under
                        development.
    transform_shocks:   True or False. True to make the rho = 1 response to
                        shock 2 be transitory.
    title:              Title for the image plotted.
    """
    colors = ['blue', 'green', 'red', 'black', 'cyan', 'magenta', 'yellow', 'black']
    mult_fac = len(rhos) // len(colors) + 1
    colors = colors * mult_fac

    smin = 0
    smax = 0
    kmin = 0
    kmax = 0
    cmin = 0
    cmax = 0
    dmin = 0
    dmax = 0
    imin = 0
    imax = 0

    fig = make_subplots(3, 2, print_grid = False, specs=[[{}, {}], [{}, {}], [{'colspan': 2}, None]])

    rtable = []
    ktable = []
    ctable = []
    itable = []
    qtable = []

    for i, r in enumerate(rhos):

        model = stochastic_growth_model(r, phi1, phi2, a_k, A, delta, beta1, beta2)
        model.find_steady_state()
        model.solve_model()

        for j, gamma in enumerate(gammas):

            model.gen_impulse_response(shock, T, gamma, B, sigk, sigd)
            S = model.S_response
            K = model.K_response
            C = model.C_response
            I = model.I_response

            if gamma == gammas[0]:
                rtable.append(model.rho)
                ktable.append(model.kstar * 4)
                ctable.append(np.exp(model.cstar) * 4)
                itable.append(np.exp(model.istar) * 4)
                qtable.append(np.exp(model.vstar * (1 - r)) * np.exp(model.cstar * r) / (1 - np.exp(-delta)))

            dmin = min(dmin, np.min(C - K) * 1.2)
            dmax = max(dmax, np.max(C - K) * 1.2)
            smin = min(smin, np.min(S) * 0.012)
            smax = max(smax, np.max(S) * 0.012)
            kmin = min(kmin, np.min(K) * 1.2)
            kmax = max(kmax, np.max(K) * 1.2)
            cmin = min(cmin, np.min(C) * 1.2)
            cmax = max(cmax, np.max(C) * 1.2)
            imin = min(imin, np.min(I - K) * 1.2)
            imax = max(imax, np.max(I - K) * 1.2)

            fig.add_scatter(y = C, row = 1, col = 1, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
            fig.add_scatter(y = K, row = 1, col = 2, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
            fig.add_scatter(y = C - K, row = 2, col = 1, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
            fig.add_scatter(y = I - K, row = 2, col = 2, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
            fig.add_scatter(y = S / 100., row = 3, col = 1, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))

    df = pd.DataFrame({"Rho":rtable, "Log capital growth":ktable,
                       "Consumption to Capital":ctable, "Investment to Capital":itable,
                       "Q":qtable})

    steps = []
    for i in range(len(gammas)):
        step = dict(
            method = 'restyle',
            args = ['visible', ['legendonly'] * len(fig.data)],
            label = 'γ = '+'{}'.format(round(gammas[i], 2))
        )
        for j in range(5):
            for k in range(len(rhos)):
                step['args'][1][i * 5 + j + k * len(gammas) * 5] = True
        steps.append(step)

    sliders = [dict(
        steps = steps
    )]

    fig.layout.sliders = sliders
    fig['layout'].update(height=800, width=1000,
                     title=title.format(shock), showlegend = False)

    fig['layout']['xaxis1'].update(range = [0, T])
    fig['layout']['xaxis2'].update(range = [0, T])
    fig['layout']['xaxis3'].update(range = [0, T])
    fig['layout']['xaxis4'].update(range = [0, T])#, title='Time (Quarters)')
    fig['layout']['xaxis5'].update(range = [0, T])

    fig['layout']['yaxis1'].update(title='Consumption', range = [cmin, cmax])
    fig['layout']['yaxis2'].update(title='Capital', range=[kmin, kmax])
    fig['layout']['yaxis3'].update(title='Consumption to Capital', range = [dmin, dmax])#showgrid=False)
    fig['layout']['yaxis4'].update(title='Investment to Capital', range = [imin, imax])
    fig['layout']['yaxis5'].update(title='Price Elasticity', range = [smin, smax])

    return df, fig


def plot_impulse_stationary(rhos, gamma, T, phi1, phi2, a_k, A, delta, beta1, beta2,
                 B, sigd, sigk, shock = 1, title = None):
    """
    Given a set of parameters, computes and displays the impulse responses of
    consumption, capital, the consumption-investment ratio, along with the
    shock price elacticities.

    Input
    ==========
    Note that the values of delta, phi, A, and a_k are specified within the code
    and are only used for the empirical_method = 0 or 0.5 specifications (see below).

    rhos:               The set of rho values for which to plot the IRFs.
    gamma:              The risk aversion of the model.
    betaz:              Shock persistence.
    T:                  Number of periods to plot.
    shock:              (1 or 2) Defines which of the two possible shocks to plot.
    empirical method:   Use 0 to use Eberly and Wang parameters and 0.5 for parameters
                        from a low adjustment cost setting. Further cases still under
                        development.
    transform_shocks:   True or False. True to make the rho = 1 response to
                        shock 2 be transitory.
    title:              Title for the image plotted.
    """
    colors = ['blue', 'green', 'red', 'black', 'cyan', 'magenta', 'yellow', 'black']
    mult_fac = len(rhos) // len(colors) + 1
    colors = colors * mult_fac

    smin = 0
    smax = 0
    kmin = 0
    kmax = 0
    cmin = 0
    cmax = 0
    dmin = 0
    dmax = 0
    imin = 0
    imax = 0

    fig = make_subplots(3, 2, print_grid = False, specs=[[{}, {}], [{}, {}], [{'colspan': 2}, None]])

    rtable = []
    ktable = []
    ctable = []
    itable = []
    qtable = []

    for i, r in enumerate(rhos):

        model = stochastic_growth_model(r, phi1, phi2, a_k, A, delta, beta1, beta2)
        model.find_steady_state()
        model.solve_model()
        model.gen_impulse_response(shock, T, gamma, B, sigk, sigd)
        S = model.S_response
        K = model.K_response
        C = model.C_response
        I = model.I_response

        dmin = min(dmin, np.min(C - K) * 1.2)
        dmax = max(dmax, np.max(C - K) * 1.2)
        smin = min(smin, np.min(S) * 0.012)
        smax = max(smax, np.max(S) * 0.012)
        kmin = min(kmin, np.min(K) * 1.2)
        kmax = max(kmax, np.max(K) * 1.2)
        cmin = min(cmin, np.min(C) * 1.2)
        cmax = max(cmax, np.max(C) * 1.2)
        imin = min(imin, np.min(I - K) * 1.2)
        imax = max(imax, np.max(I - K) * 1.2)

        fig.add_scatter(y = C, row = 1, col = 1, visible = True,
                        name = 'rho = {}'.format(r),
                        line = dict(color = (colors[i])), showlegend = False)
        fig.add_scatter(y = K, row = 1, col = 2, visible = True,
                        name = 'rho = {}'.format(r),
                        line = dict(color = (colors[i])), showlegend = False)
        fig.add_scatter(y = C - K, row = 2, col = 1, visible = True,
                        name = 'rho = {}'.format(r),
                        line = dict(color = (colors[i])), showlegend = False)
        fig.add_scatter(y = I - K, row = 2, col = 2, visible = True,
                        name = 'rho = {}'.format(r),
                        line = dict(color = (colors[i])), showlegend = False)
        fig.add_scatter(y = S / 100., row = 3, col = 1, visible = True,
                        name = 'rho = {}'.format(r),
                        line = dict(color = (colors[i])))

    fig['layout']['xaxis1'].update(range = [0, T])
    fig['layout']['xaxis2'].update(range = [0, T])
    fig['layout']['xaxis3'].update(range = [0, T])
    fig['layout']['xaxis4'].update(range = [0, T])#, title='Time (Quarters)')
    fig['layout']['xaxis5'].update(range = [0, T])

    fig['layout']['yaxis1'].update(title='Consumption', range = [cmin, cmax])
    fig['layout']['yaxis2'].update(title='Capital', range=[kmin, kmax])
    fig['layout']['yaxis3'].update(title='Consumption to Capital', range = [dmin, dmax])#showgrid=False)
    fig['layout']['yaxis4'].update(title='Investment to Capital', range = [imin, imax])
    fig['layout']['yaxis5'].update(title='Price Elasticity', range = [smin, smax])

    fig['layout']['width'] = 1000
    fig['layout']['height'] = 700
    fig['layout']['title'] = title.format(shock)

    return fig

if __name__ == "__main__":
    # Can be used for simple disgnostics
    r = 1
    phi2 = 100
    model = stochastic_growth_model(r, 5./phi2, phi2, 0.025, 0.036, 0.005, 0.014, 0.0022)
    model.find_steady_state()
    model.solve_model()
