import numpy as np
from types import SimpleNamespace
from scipy.optimize import minimize

class MarketClearing:
    def __init__(self, A, gamma, alpha, nu, epsilon, tau, T, w):
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.nu = nu
        self.epsilon = epsilon
        self.tau = tau
        self.T = T
        self.w = w
        self.kappa = 0.1
        self.p1_values = np.linspace(0.1, 2.0, 10)
        self.p2_values = np.linspace(0.1, 2.0, 10)
        self.X0 = np.array([0.1, 0.1])  # Initial guess for tau and T
    
    def optimal_labor_supply(self, p):
        return (p * self.A * self.gamma / self.w) ** (1 / (1 - self.gamma))

    def optimal_output(self, labor):
        return self.A * labor ** self.gamma

    def implied_profits(self, p):
        labor = self.optimal_labor_supply(p)
        return (1 - self.gamma) / self.gamma * self.w * labor

    def utility(self, c1, c2, ell):
        return np.log(c1 ** self.alpha * c2 ** (1 - self.alpha)) - self.nu * ell ** (1 + self.epsilon) / (1 + self.epsilon)

    def check_market_clearing(self, par):
        results = []
        for p1 in self.p1_values:
            for p2 in self.p2_values:
                ell1_star = self.optimal_labor_supply(p1)
                ell2_star = self.optimal_labor_supply(p2)
                ell_star = ell1_star + ell2_star

                y1_star = self.optimal_output(ell1_star)
                y2_star = self.optimal_output(ell2_star)

                pi1_star = self.implied_profits(p1)
                pi2_star = self.implied_profits(p2)

                income = self.w * ell_star + self.T + pi1_star + pi2_star
                c1_star = self.alpha * income / p1
                c2_star = (1 - self.alpha) * income / (p2 + self.tau)

                labor_clearing = np.abs(ell_star - (ell1_star + ell2_star)) < 1e-6
                good1_clearing = np.abs(c1_star - y1_star) < 1e-6
                good2_clearing = np.abs(c2_star - y2_star) < 1e-6

                results.append((p1, p2, labor_clearing, good1_clearing, good2_clearing))
        return results

    def print_results(self, results):
        for result in results:
            p1, p2, labor_clearing, good1_clearing, good2_clearing = result
            print(f"p1: {p1:.2f}, p2: {p2:.2f} | Labor Clearing: {labor_clearing} | Good 1 Clearing: {good1_clearing} | Good 2 Clearing: {good2_clearing}")
     
    def market_clearing(self, prices, par):
        p1, p2 = prices
        
        ell1_star = self.optimal_labor_supply(p1)
        ell2_star = self.optimal_labor_supply(p2)
        ell_star = ell1_star + ell2_star

        y1_star = self.optimal_output(ell1_star)
        y2_star = self.optimal_output(ell2_star)

        pi1_star = self.implied_profits(p1)
        pi2_star = self.implied_profits(p2)

        income = self.w * ell_star + self.T + pi1_star + pi2_star
        c1_star = self.alpha * income / p1
        c2_star = (1 - self.alpha) * income / (p2 + self.tau)

        good1_market = y1_star - c1_star
        good2_market = y2_star - c2_star

        return [good1_market, good2_market]

    def objective(self, prices, par):
        return sum(np.abs(self.market_clearing(prices, par)))

    def solve_equilibrium(self, par):
        initial_guess = [1.0, 1.0]
        bounds = [(0.1, 2.0), (0.1, 2.0)]
        
        result = minimize(self.objective, initial_guess, args=(par,), bounds=bounds, method='L-BFGS-B')
        return result.x        
    
    def calculate_equilibrium_values(self, prices, par):
        p1, p2 = prices

        ell1_star = self.optimal_labor_supply(p1)
        ell2_star = self.optimal_labor_supply(p2)
        ell_star = ell1_star + ell2_star

        y1_star = self.optimal_output(ell1_star)
        y2_star = self.optimal_output(ell2_star)

        pi1_star = self.implied_profits(p1)
        pi2_star = self.implied_profits(p2)

        income = self.w * ell_star + self.T + pi1_star + pi2_star
        c1_star = self.alpha * income / p1
        c2_star = (1 - self.alpha) * income / (p2 + self.tau)

        self.T = self.tau * c2_star

        labor_clearing = np.abs(ell_star - (ell1_star + ell2_star)) < 1e-6
        good1_clearing = np.abs(c1_star - y1_star) < 1e-6
        good2_clearing = np.abs(c2_star - y2_star) < 1e-6

        return {
            'ell1_star': ell1_star,
            'ell2_star': ell2_star,
            'ell_star': ell_star,
            'y1_star': y1_star,
            'y2_star': y2_star,
            'pi1_star': pi1_star,
            'pi2_star': pi2_star,
            'income': income,
            'c1_star': c1_star,
            'c2_star': c2_star,
            'labor_clearing': labor_clearing,
            'good1_clearing': good1_clearing,
            'good2_clearing': good2_clearing
        }


    # def objective_SWF(self, params):
    #     tau, T = params
        
    #     # Set current values of tau and T
    #     self.tau = tau
    #     self.T = T
        
    #     # Calculate equilibrium prices
    #     prices = self.solve_equilibrium(par=(self.tau, self.T, self.w))
        
    #     # Calculate equilibrium values
    #     equilibrium_values = self.calculate_equilibrium_values(prices)
        
    #     # Extract relevant values
    #     U = self.utility(equilibrium_values['c1_star'], equilibrium_values['c2_star'], equilibrium_values['ell_star'])
    #     y2_star = equilibrium_values['y2_star']
        
    #     # Calculate SWF
    #     SWF = U - self.kappa * y2_star
        
    #     # Maximize SWF (minimize -SWF)
    #     return -SWF

    # def maximize_SWF(self):
    #     initial_guess = [0.0, 0.0]  # Initial guess for tau and T
    #     bounds = [(0.0, None), (0.0, None)]  # Non-negative bounds for tau and T
        
    #     result = minimize(self.objective_SWF, initial_guess, bounds=bounds, method='L-BFGS-B')
        
    #     # Extract optimal tau and T
    #     optimal_tau, optimal_T = result.x
        
    #     # Update tau and T in the class
    #     self.tau = optimal_tau
    #     self.T = optimal_T
        
    #     # Print or return results
    #     print(f"Optimal tau: {optimal_tau}")
    #     print(f"Implied T: {self.T}")
    #     print(f"Maximized SWF: {-result.fun}")  # -result.fun because we minimized -SWF
        
    #     return optimal_tau, optimal_T
    


    
    # def objective_function(self, params):
    #     tau, T = params
        
    #     # Calculate equilibrium prices
    #     prices = self.solve_equilibrium(par=(tau, T, self.w))
        
    #     # Calculate market clearing conditions
    #     goods_market = self.market_clearing(prices, par=(tau, T))
        
    #     # Objective function to minimize (sum of squares)
    #     return np.sum(np.array(goods_market) ** 2)

    # def solve_continuous(self):
    #     """ Continuous optimization for tau and T """
    #     bounds = [(0.0, 10), (0.0, 10)]  # Non-negative bounds for tau and T
    #     result = minimize(self.objective_function, self.X0, bounds=bounds, method='SLSQP')
    #     optimal_tau, optimal_T = result.x
        
    #     print(f"Optimal tau: {optimal_tau}")
    #     print(f"Optimal T: {optimal_T}")
        
    #     # Update self.tau and self.T with optimal values if needed
    #     self.tau = optimal_tau
    #     self.T = optimal_T
        
    #     return optimal_tau, optimal_T



    def market_clearing(self, prices):
        tau, T = prices
        print(f"Current tau: {tau}, Current T: {T}")
        
        p1 = 1.0  # Placeholder, adjust based on your model
        p2 = 1.0  # Placeholder, adjust based on your model
        
        ell1_star = self.optimal_labor_supply(p1)
        ell2_star = self.optimal_labor_supply(p2)
        ell_star = ell1_star + ell2_star

        y1_star = self.optimal_output(ell1_star)
        y2_star = self.optimal_output(ell2_star)

        pi1_star = self.implied_profits(p1)
        pi2_star = self.implied_profits(p2)

        income = self.w * ell_star + T + pi1_star + pi2_star
        c1_star = self.alpha * income / p1
        c2_star = (1 - self.alpha) * income / (p2 + tau)

        good1_market = y1_star - c1_star
        good2_market = y2_star - c2_star

        print(f"good1_market: {good1_market}, good2_market: {good2_market}")

        return [good1_market, good2_market]

    def objective_function(self, params):
        tau, T = params
        
        # Calculate market clearing conditions
        goods_market = self.market_clearing((tau, T))
        
        # Objective function to minimize (sum of squares)
        objective_value = np.sum(np.array(goods_market) ** 2)
        print(f"Objective value: {objective_value} for tau: {tau}, T: {T}")
        return objective_value

    def solve_continuous(self):
        """ Continuous optimization for tau and T """
        bounds = [(0.0, None), (0.0, None)]  # Non-negative bounds for tau and T
        result = minimize(self.objective_function, self.X0, bounds=bounds, method='SLSQP')
        optimal_tau, optimal_T = result.x
        
        print(f"Optimal tau: {optimal_tau}")
        print(f"Optimal T: {optimal_T}")
        
        # Update self.tau and self.T with optimal values if needed
        self.tau = optimal_tau
        self.T = optimal_T
        
        return optimal_tau, optimal_T