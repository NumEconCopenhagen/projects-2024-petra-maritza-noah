# import numpy as np
# from scipy.optimize import minimize

# class MarketClearing:
#     def __init__(self, A=1, gamma=0.5, alpha=0.5, nu=1, epsilon=1, w=1):
#         self.A = A
#         self.gamma = gamma
#         self.alpha = alpha
#         self.nu = nu
#         self.epsilon = epsilon
#         self.w = w

#     def labor_demand(self, p):
#         return (p * self.A * self.gamma / self.w) ** (1 / (1 - self.gamma))

#     def output(self, l):
#         return self.A * l ** self.gamma

#     def profit(self, p):
#         return (1 - self.gamma) / self.gamma * self.w * (p * self.A * self.gamma / self.w) ** (1 / (1 - self.gamma))

#     def c1(self, l, T, pi1, pi2, p1):
#         return self.alpha * (self.w * l + T + pi1 + pi2) / p1

#     def c2(self, l, T, pi1, pi2, p2, tau):
#         return (1 - self.alpha) * (self.w * l + T + pi1 + pi2) / (p2 + tau)

#     def utility(self, l, T, pi1, pi2, p1, p2, tau):
#         c1_val = self.c1(l, T, pi1, pi2, p1)
#         c2_val = self.c2(l, T, pi1, pi2, p2, tau)
#         return np.log(c1_val ** self.alpha * c2_val ** (1 - self.alpha)) - self.nu * (l ** (1 + self.epsilon)) / (1 + self.epsilon)

#     def find_optimal_labor_supply(self, p1, p2, tau=0, T=0):
#         pi1_star = self.profit(p1)
#         pi2_star = self.profit(p2)
        
#         result = minimize(lambda l: -self.utility(l, T, pi1_star, pi2_star, p1, p2, tau), 
#                           x0=1, bounds=[(0.01, 10)])
#         if result.success:
#             return result.x[0]
#         else:
#             print(f"Optimization failed for p1={p1}, p2={p2}")
#             return None

#     def market_clearing_conditions(self, prices):
#         p1, p2 = prices
#         l1_star = self.labor_demand(p1)
#         l2_star = self.labor_demand(p2)
#         y1_star = self.output(l1_star)
#         y2_star = self.output(l2_star)
#         pi1_star = self.profit(p1)
#         pi2_star = self.profit(p2)

#         T = 0
#         tau = 0

#         l_star = self.find_optimal_labor_supply(p1, p2, tau, T)

#         if l_star is not None:
#             c1_star = self.c1(l_star, T, pi1_star, pi2_star, p1)
#             c2_star = self.c2(l_star, T, pi1_star, pi2_star, p2, tau)
            
#             labor_market_clears = np.isclose(l_star, l1_star + l2_star)
#             good1_market_clears = np.isclose(c1_star, y1_star)
#             good2_market_clears = np.isclose(c2_star, y2_star)
            
#             return (not labor_market_clears) + (not good1_market_clears) + (not good2_market_clears)
#         else:
#             return float('inf')

#     def find_market_clearing_prices(self):
#         result = minimize(self.market_clearing_conditions, x0=[1, 1], bounds=[(0.1, 10), (0.1, 10)])
#         if result.success:
#             return result.x
#         else:
#             print("Optimization failed to find market-clearing prices.")
#             return None 
        

import numpy as np
from scipy.optimize import minimize

class MarketClearing:
    def __init__(self, A=1, gamma=0.5, alpha=0.5, nu=1, epsilon=1, w=1):
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.nu = nu
        self.epsilon = epsilon
        self.w = w

    def labor_demand(self, p):
        return (p * self.A * self.gamma / self.w) ** (1 / (1 - self.gamma))

    def output(self, l):
        return self.A * l ** self.gamma

    def profit(self, p):
        return (1 - self.gamma) / self.gamma * self.w * (p * self.A * self.gamma / self.w) ** (1 / (1 - self.gamma))

    def c1(self, l, T, pi1, pi2, p1):
        return self.alpha * (self.w * l + T + pi1 + pi2) / p1

    def c2(self, l, T, pi1, pi2, p2, tau):
        return (1 - self.alpha) * (self.w * l + T + pi1 + pi2) / (p2 + tau)

    def utility(self, l, T, pi1, pi2, p1, p2, tau):
        c1_val = self.c1(l, T, pi1, pi2, p1)
        c2_val = self.c2(l, T, pi1, pi2, p2, tau)
        return np.log(c1_val ** self.alpha * c2_val ** (1 - self.alpha)) - self.nu * (l ** (1 + self.epsilon)) / (1 + self.epsilon)

    def find_optimal_labor_supply(self, p1, p2, tau=0, T=0):
        pi1_star = self.profit(p1)
        pi2_star = self.profit(p2)
        
        result = minimize(lambda l: -self.utility(l, T, pi1_star, pi2_star, p1, p2, tau), 
                          x0=1, bounds=[(0.01, 10)])
        if result.success:
            return result.x[0]
        else:
            print(f"Optimization failed for p1={p1}, p2={p2}")
            return None

    def check_market_clearing(self, p1_grid, p2_grid):
        results = []
        for p1 in p1_grid:
            for p2 in p2_grid:
                l1_star = self.labor_demand(p1)
                l2_star = self.labor_demand(p2)
                y1_star = self.output(l1_star)
                y2_star = self.output(l2_star)
                pi1_star = self.profit(p1)
                pi2_star = self.profit(p2)
                
                T = 0
                tau = 0
                
                l_star = self.find_optimal_labor_supply(p1, p2, tau, T)
                
                if l_star is not None:
                    c1_star = self.c1(l_star, T, pi1_star, pi2_star, p1)
                    c2_star = self.c2(l_star, T, pi1_star, pi2_star, p2, tau)
                    
                    labor_market_clears = np.isclose(l_star, l1_star + l2_star)
                    good1_market_clears = np.isclose(c1_star, y1_star)
                    good2_market_clears = np.isclose(c2_star, y2_star)
                    
                    results.append((p1, p2, labor_market_clears, good1_market_clears, good2_market_clears))
                else:
                    results.append((p1, p2, False, False, False))
        
        return results

    def find_market_clearing_prices(self):
        p1_grid = np.linspace(0.1, 2.0, 10)
        p2_grid = np.linspace(0.1, 2.0, 10)
        
        results = self.check_market_clearing(p1_grid, p2_grid)
        
        for result in results:
            p1, p2, labor_market_clears, good1_market_clears, good2_market_clears = result
            if labor_market_clears and good1_market_clears and good2_market_clears:
                return p1, p2
        
        print("Market clearing prices not found within the given grids.")
        return None, None