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
        
        Y1: Income of the children
        I1: Luckiness
        H1: Human capital level
        H1 depends on:
        X0: Parents' investment in education
        S0: Governments' investment in educationù
        E1: Endowment (intergenerational feature)
        r: Financial market rate at which parent can borrow money
        D: Debt that the children inherit
        """

        # a. create namespaces
        par = self.par = SimpleNamespace() #group of variables that now is empty
        sol = self.sol = SimpleNamespace()

        #b.parameters influencing decision making
        par.Y1 = 0 #Placeholder for income, to be determined 
        par.I1 = 1
        par.H1 = 0 # Placeholder for human capital, to be determined
        par.alpha=0.5 # Social endowment
        par.vt=0.6 # luck, unsystematic component
        par.h= 0.8 # degree of inheritability of endowments (0<h<1)
        par.X0 = 0.5  # Initial guess for parental investment
        par.S0 = 1.0  # Government investment in education
        par.E0 = 2    # Initial endowment
        par.rt = 0.05 # Financial market rate (borrowing rate)
        par.Y0 = 5 # Parents'income in the first period
        par.C0 = 5 #Parents'consumption in the first period

    def human_capital_production(self, X0, S0, E1):
        """ Human capital production function H1 = f(X0, S0, E1) """
        return X0**0.5 + S0**0.3 + 0.1*E1  # Example functional form

    def endowment_production(self, E0):
        """ Endowment production function E1 = alpha + h*E0 + vt """
        return self.par.alpha + self.par.h * E0 + self.par.vt

    def calc_income(self, H1, I1):
        """ Calculate total income Y1 = H1 + I1 """
        return H1 + I1
    
    def calc_debt(self, X0, Y0):
        # Calculate debt based on X and Y
        return self.par.C0 + self.par.X0 - self.par.Y0

    def objective_function(self, X0):
        """ Objective function to maximize (here to be minimized by scipy) """
        E1 = self.endowment_production(self.par.E0)
        H1 = self.human_capital_production(X0, self.par.S0, E1)
        Y1 = self.calc_income(H1, self.par.I1)
        D1 = self.calc_debt(X0, self.par.Y0)

        objective_value = Y1 - (1 + self.par.rt) * D1
        # We negate the income as we'll use a minimizer to maximize it
        return -objective_value


    def constraint_function(self, X):
        # Calculate the marginal rate of investment return
        epsilon = 1e-6  # Small value for numerical stability
        r_m = (self.objective_function(X + epsilon) - self.objective_function(X)) / epsilon
        
        # Constraint: r_m - r = 0
        return r_m - self.par.rt

    
    def solve_continuous(self):
        # Initial guess for X
        X0 = self.par.X0
        
        # Optimization with constraint
        result = minimize(self.objective_function, X0, constraints={'type': 'eq', 'fun': self.constraint_function})
        
        # Extract optimal X value
        optimal_X = result.x
        
        # Calculate optimal H, Y, and D based on optimal X
        optimal_H = self.human_capital_production(optimal_X, self.par.S0, self.par.E0)
        optimal_Y = self.calc_income(optimal_H, self.par.I1)
        optimal_D = self.calc_debt(optimal_X, optimal_Y)
        
        # Return optimal values
        return {'optimal_X': optimal_X, 'optimal_H': optimal_H, 'optimal_Y': optimal_Y, 'optimal_D': optimal_D}




    # def solve_continuous(self, do_print=False):
    #     """Solve model continuously"""

    #     par = self.par
    #     sol = self.sol
    #     opt = SimpleNamespace()

    #     #a. calculate income with negative since we will use minimize()
    #     def objective(x):
    #         return self.objective_function(x[0])
        
    #     #b. constraints and bounds
    #     bounds = [(0, None)]

    #     #c. initial guess
    #     x_guess = np.array([0.5])

    #     #d. find maximization
    #     result = minimize(objective, x_guess, method="SLSQP", bounds=bounds)

    # # Store optimal solution
    #     opt.X0_optimal = result.x[0]
    #     opt.E1 = self.endowment_production(par.E0)
    #     opt.H1_optimal = self.human_capital_production(opt.X0_optimal, par.S0, opt.E1)
    #     opt.Y1_optimal = self.calc_income(opt.H1_optimal, par.I1)

    # # Print answer if required
    #     if do_print:
    #         print(f'Optimal parental investment: {opt.X0_optimal:6.4f}')
    #         print(f'Human capital level: {opt.H1_optimal:6.4f}')
    #         print(f'Optimal income: {opt.Y1_optimal:6.4f}')

    #     return opt



#from here solving the model continuously and discretely:
    # def solve_continuous(self):
    #     """ Solve the model continuously using scipy's optimization routines """
    #     result = minimize(self.objective_function, self.par.X0, bounds=[(0, None)])
    #     self.sol.X0_optimal = result.x[0]
    #     self.sol.E1 = self.endowment_production(self.par.E0)
    #     self.sol.H1_optimal = self.human_capital_production(self.sol.X0_optimal, self.par.S0, self.sol.E1)
    #     self.sol.Y1_optimal = self.calc_income(self.sol.H1_optimal, self.par.I1)
    #     return self.sol
    

    # def solve_continuous(self):
    #     """ Solve the model continuously using scipy's optimization routines """
    #     # Provide a better initial guess
    #     initial_guess = 0.1  # Adjust as needed
        
    #     # Tighter bounds based on the feasible range of X0
    #     lower_bound = 0.0
    #     upper_bound = 1.0
        
    #     # Adjust optimization algorithm and increase max iterations
    #     result = minimize(self.objective_function, initial_guess, 
    #                       bounds=[(lower_bound, upper_bound)],
    #                       method='L-BFGS-B',  # Another optimization method to try
    #                       options={'maxiter': 1000})  # Increase max iterations
        
    #     self.sol.X0_optimal = result.x[0]
    #     self.sol.E1 = self.endowment_production(self.par.E0)
    #     self.sol.H1_optimal = self.human_capital_production(self.sol.X0_optimal, self.par.S0, self.sol.E1)
    #     self.sol.Y1_optimal = self.calc_income(self.sol.H1_optimal, self.par.I1)
    #     return self.sol
    


    # def solve_discrete(self):
    #     """ Solve the model discretely """
    #     # Generate all possible choices for X0
    #     X0_values = np.linspace(0, 1, num=100)
    #     max_income = -np.inf
    #     optimal_X0 = None

    #     # Iterate through all possible X0 values
    #     for X0 in X0_values:
    #         E1 = self.endowment_production(self.par.E0)
    #         H1 = self.human_capital_production(X0, self.par.S0, E1)
    #         Y1 = self.calc_income(H1, self.par.I1)

    #         # Update optimal solution if income is higher
    #         if Y1 > max_income:
    #             max_income = Y1
    #             optimal_X0 = X0

    #     # Calculate optimal values for other variables based on the optimal X0
    #     self.sol.X0_optimal = optimal_X0
    #     self.sol.E1 = self.endowment_production(self.par.E0)
    #     self.sol.H1_optimal = self.human_capital_production(optimal_X0, self.par.S0, self.sol.E1)
    #     self.sol.Y1_optimal = max_income
    #     return self.sol

    def solve_discrete(self):
        """ Solve the model discretely """
        # Generate all possible choices for X0
        X0_values = np.linspace(0, 1, num=100)
        max_utility = -np.inf
        optimal_X0 = None

        # Iterate through all possible X0 values
        for X0 in X0_values:
        # Calculate utility for the current X0
            utility = self.objective_function(X0)

        # Update optimal solution if utility is higher
            if utility > max_utility:
                max_utility = utility
                optimal_X0 = X0

    # Calculate optimal values for other variables based on the optimal X0
        optimal_E1 = self.endowment_production(self.par.E0)
        optimal_H1 = self.human_capital_production(optimal_X0, self.par.S0, optimal_E1)
        optimal_Y1 = self.calc_income(optimal_H1, self.par.I1)
        optimal_D1 = self.calc_debt(optimal_X0, optimal_Y1)

    # Store optimal values in the solution namespace
        self.sol.optimal_X0 = optimal_X0
        self.sol.optimal_E1 = optimal_E1
        self.sol.optimal_H1 = optimal_H1
        self.sol.optimal_Y1 = optimal_Y1
        self.sol.optimal_D1 = optimal_D1

        return self.sol

