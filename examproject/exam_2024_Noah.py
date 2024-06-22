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
        self.p1_values = np.linspace(0.1, 2.0, 10)
        self.p2_values = np.linspace(0.1, 2.0, 10)
    
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




