import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy.optimize import minimize
import math

class BeckerTomesModelDebugged:

    def __init__(self):
        """ Setup model of parents' private investment in education, with the aim of maximizing the income of children."""
        # Parameters
        self.par = SimpleNamespace(
            I1=0.5,           # Luckiness, set to somewhat of an important factor given the value
            H1=0,             # Placeholder for human capital
            alpha=0.5,        # Social endowment
            vt=0.6,           # Luck, unsystematic component
            h=0.8,            # Degree of inheritability of endowments
            X0=0.5,           # Initial guess for parental investment
            S0=1.0,           # Government investment in education
            E0=2,             # Initial endowment
            rt=0.05,          # Financial market rate (borrowing rate)
            Y0=5,             # Parents' income in the first period. Our model is not taking in account credit constraints, hence income and consumption are set equal.
            C0=5              # Parents' consumption in the first period
        )
        self.sol = SimpleNamespace()

    def human_capital_production(self, X0, S0, E1):
        """ Human capital production function """
        H1 = X0**0.4 + S0**0.5 + 0.3*E1
        #print(f"Calculating Human Capital: H1={H1} for X0={X0}, S0={S0}, E1={E1}")
        return H1

    def endowment_production(self, E0):
        """ Endowment production function """
        E1 = self.par.alpha + self.par.h * E0 + self.par.vt
        #print(f"Calculating Endowment: E1={E1} for E0={E0}")
        return E1

    def calc_income(self, H1, I1):
        """ Calculate total income """
        Y1 = H1 + I1
        #print(f"Calculating Income: Y1={Y1} for H1={H1}, I1={I1}")
        return Y1

    def calc_debt(self, X0, Y0):
        """ Calculate debt """
        D1 = self.par.C0 + X0 - Y0
        #print(f"Calculating Debt: D1={D1} for X0={X0}, Y0={Y0}")
        return D1

    def objective_function(self, X0):
        """ Objective function to maximize (negative for minimization) """
        E1 = self.endowment_production(self.par.E0)
        H1 = self.human_capital_production(X0, self.par.S0, E1)
        Y1 = self.calc_income(H1, self.par.I1)
        D1 = self.calc_debt(X0, self.par.Y0)
        obj_val = -(Y1 - (1 + self.par.rt) * D1)
        #print(f"Objective Function: Value={obj_val} for X0={X0}")
        return obj_val

    def solve_continuous(self):
        """ Continuous optimization """
        result = minimize(self.objective_function, self.par.X0, method='SLSQP')
        self.sol.optimal_X = result.x[0]
        print(f"Continuous Solution: Optimal X0={self.sol.optimal_X}")

    def solve_discrete(self):
        """ Discrete optimization """
        X0_values = np.linspace(0, 10, 100)  # Expanded search range for comparison
        max_utility = -np.inf
        optimal_X0 = None

        for X0 in X0_values:
            utility = -self.objective_function(X0)  # Negate to maximize
            if utility > max_utility:
                max_utility = utility
                optimal_X0 = X0
                #print(f"New Optimal Found: X0={X0}, Utility={utility}")

        self.sol.optimal_X0 = optimal_X0
        print(f"Discrete Solution: Optimal X0={self.sol.optimal_X0}")

# Create an instance of the model and solve it
model = BeckerTomesModelDebugged()
model.solve_continuous()
model.solve_discrete()