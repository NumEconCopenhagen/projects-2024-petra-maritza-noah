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