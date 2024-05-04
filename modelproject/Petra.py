import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy import optimize
from scipy.optimize import minimize
import math

class BeckerTomesModelClass:

    def __init__(self):
        """ Setup model of parents' private investment in education, with the aim of maximizing the income of children.

        The model has two periods and a representative individual.
        
        The model depend on the following parameters:
        
        Yt: Income of the children
        It: Luckiness
        Ht: Human capital level
        Ht depends on:
        Xt-1: Parents' investment in education
        St-1: Governments' investment in educationù
        Et: Endowment (intergenerational feature)
        rt: Financial market rate at which parent can borrow money
        Dt-1: Debt that the children inherit
        """

        # a. create namespaces
        par = self.par = SimpleNamespace() #group of variables that now is empty
        sol = self.sol = SimpleNamespace()

        #b.parameters influencing decision making
        par.Y1 = 10
        par.I1 = 1
        par.H1 = 0 # Placeholder for human capital, to be determined
        par.alpha=0.5 # Social endowment
        par.vt=0.6 # luck, unsystematic component
        par.h= 0.8 # degree of inheritability of endowments (0<h<1)
        par.X0 = 0.5  # Initial guess for parental investment
        par.S0 = 1.0  # Government investment in education
        par.E0 = 2    # Initial endowment
        par.rt = 0.05 # Financial market rate (borrowing rate)

    #maybe with self i have to put self.par in every par that i have defined
    def human_capital_production(self, X0, S0, E1):
        """ Human capital production function H_t = f(X0, S0, E1) """
        return X0**0.5 + S0**0.3 + 0.1*E1  # Example functional form

    def endowment_production(self, E0):
        """ Endowment production function E_t = alpha + h*E_t-1 + vt """
        return self.par.alpha + self.par.h * E0 + self.par.vt

    def calc_income(self, H1, I1):
        """ Calculate total income Y_t = H_t + I_t """
        return H1 + I1

    def objective_function(self, X0):
        """ Objective function to maximize (here to be minimized by scipy) """
        E1 = self.endowment_production(self.par.E0)
        H1 = self.human_capital_production(X0, self.par.S0, E1)
        Y1 = self.calc_income(H1, self.par.I1)
        # We negate the income as we'll use a minimizer to maximize it
        return -Y1

    def solve_model(self):
        """ Solve the model using scipy's optimization routines """
        result = minimize(self.objective_function, self.par.X0, bounds=[(0, None)])
        self.sol.X0_optimal = result.x[0]
        self.sol.E1 = self.endowment_production(self.par.E0)
        self.sol.H1_optimal = self.human_capital_production(self.sol.X0_optimal, self.par.S0, self.sol.E1)
        self.sol.Y1_optimal = self.calc_income(self.sol.H1_optimal, self.par.I1)

        return self.sol

 # Usage for the notebook
# model = BeckerTomesModelClass()
# solution = model.solve_model()
# print(f'Optimal parental investment: {solution.X0_optimal}')
# print(f'Human capital level: {solution.H1_optimal}')
# print(f'Optimal income: {solution.Y1_optimal}')
