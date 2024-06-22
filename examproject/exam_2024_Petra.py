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

#     def check_market_clearing(self, p1_grid, p2_grid):
#         results = []
#         for p1 in p1_grid:
#             for p2 in p2_grid:
#                 l1_star = self.labor_demand(p1)
#                 l2_star = self.labor_demand(p2)
#                 y1_star = self.output(l1_star)
#                 y2_star = self.output(l2_star)
#                 pi1_star = self.profit(p1)
#                 pi2_star = self.profit(p2)
                
#                 T = 0
#                 tau = 0
                
#                 l_star = self.find_optimal_labor_supply(p1, p2, tau, T)
                
#                 if l_star is not None:
#                     c1_star = self.c1(l_star, T, pi1_star, pi2_star, p1)
#                     c2_star = self.c2(l_star, T, pi1_star, pi2_star, p2, tau)
                    
#                     labor_market_clears = np.isclose(l_star, l1_star + l2_star)
#                     good1_market_clears = np.isclose(c1_star, y1_star)
#                     good2_market_clears = np.isclose(c2_star, y2_star)

#                     results.append((p1, p2, labor_market_clears, good1_market_clears, good2_market_clears))
#                 else:
#                     results.append((p1, p2, False, False, False))
        
#         return results

#     def find_market_clearing_prices(self):
#         p1_grid = np.linspace(0.1, 2.0, 10)
#         p2_grid = np.linspace(0.1, 2.0, 10)
        
#         results = self.check_market_clearing(p1_grid, p2_grid)
        
#         for result in results:
#             p1, p2, labor_market_clears, good1_market_clears, good2_market_clears = result
#             if labor_market_clears and good1_market_clears and good2_market_clears:
#                 return p1, p2
        
#         print("Market clearing prices not found within the given grids.")
#         return None, None



# import numpy as np
# from scipy.optimize import minimize

# class Firm:
#     def __init__(self, A=1, gamma=0.5, alpha=0.5):
#         self.A = A
#         self.gamma = gamma
#         self.alpha = alpha

#     def labor_demand(self, p, w):
#         return (p * self.A * self.gamma / w) ** (1 / (1 - self.gamma))

#     def output(self, l):
#         return self.A * l ** self.gamma

#     def profit(self, p, w):
#         return (1 - self.gamma) / self.gamma * w * (p * self.A * self.gamma / w) ** (1 / (1 - self.gamma))


# class Consumer:
#     def __init__(self, nu=1, epsilon=1, alpha=0.5):
#         self.nu = nu
#         self.epsilon = epsilon
#         self.alpha = alpha

#     def utility(self, c1, c2, l):
#         return np.log(c1 ** self.alpha * c2 ** (1 - self.alpha)) - self.nu * (l ** (1 + self.epsilon)) / (1 + self.epsilon)

#     def optimal_consumption(self, w, p1, p2, pi1_star, pi2_star, tau=0, T=0):
#         c1_star = self.alpha * (w + T + pi1_star + pi2_star) / p1
#         c2_star = (1 - self.alpha) * (w + T + pi1_star + pi2_star) / (p2 + tau)
#         return c1_star, c2_star

#     def optimal_labor_supply(self, w, p1, p2):
#         result = minimize(lambda l: -self.utility(*self.optimal_consumption(w, p1, p2, 0, 0, 0, 0), l),
#                           x0=1, bounds=[(0.01, 10)])
#         if result.success:
#             return result.x[0]
#         else:
#             print(f"Optimization failed for w={w}, p1={p1}, p2={p2}")
#             return None


# class MarketClearing:
#     def __init__(self, firm1, firm2, consumer):
#         self.firm1 = firm1
#         self.firm2 = firm2
#         self.consumer = consumer

#     def check_market_clearing(self, w, p1, p2, atol=1e-8):
#         l1_star = self.firm1.labor_demand(p1, w)
#         l2_star = self.firm2.labor_demand(p2, w)
#         y1_star = self.firm1.output(l1_star)
#         y2_star = self.firm2.output(l2_star)
#         pi1_star = self.firm1.profit(p1, w)
#         pi2_star = self.firm2.profit(p2, w)

#         l_star = self.consumer.optimal_labor_supply(w, p1, p2)
#         if l_star is None:
#             return False, False, False

#         c1_star, c2_star = self.consumer.optimal_consumption(w, p1, p2, pi1_star, pi2_star)

#         labor_market_clears = np.isclose(l_star, l1_star + l2_star, atol=atol)
#         good1_market_clears = np.isclose(c1_star, y1_star, atol=atol)
#         good2_market_clears = np.isclose(c2_star, y2_star, atol=atol)

#         return labor_market_clears, good1_market_clears, good2_market_clears


# Question 3:




# import numpy as np
# from scipy.spatial.distance import cdist

# class pointinterpolator:
#     def __init__(self, X):
#         self.X = X

#     def find_nearest_points(self, y):
#         # Calculate distances from y to all points in X
#         distances = cdist(np.array([y]), self.X)
#         indices = np.argsort(distances)[0]  # Sort indices by distance
#         sorted_X = self.X[indices]  # Sorted points in X by distance to y
#         return sorted_X[:4]  # Return the 4 closest points

#     def compute_barycentric_coordinates(self, y, A, B, C):
#         # Compute barycentric coordinates for triangle ABC
#         denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
#         r1 = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denom
#         r2 = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denom
#         r3 = 1.0 - r1 - r2
#         return r1, r2, r3

#     def interpolate_value(self, y):
#         # Find nearest points A, B, C, D
#         try:
#             A, B, C, D = self.find_nearest_points(y)
#         except IndexError:
#             print("Not enough points in X to form quadrilateral ABCD.")
#             return None

#         # Compute barycentric coordinates for triangles ABC and CDA
#         if A is not None and B is not None and C is not None and D is not None:
#             r1_ABC, r2_ABC, r3_ABC = self.compute_barycentric_coordinates(y, A, B, C)
#             r1_CDA, r2_CDA, r3_CDA = self.compute_barycentric_coordinates(y, C, D, A)

#             # Define your function here, for example:
#             def f(x1, x2):
#                 return np.sin(2*np.pi*x1) * np.cos(2*np.pi*x2)

#             # Interpolation
#             if 0 <= r1_ABC <= 1 and 0 <= r2_ABC <= 1 and 0 <= r3_ABC <= 1:
#                 interpolated_value_ABC = r1_ABC * f(A[0], A[1]) + r2_ABC * f(B[0], B[1]) + r3_ABC * f(C[0], C[1])
#                 return interpolated_value_ABC, A, B, C, D
#             elif 0 <= r1_CDA <= 1 and 0 <= r2_CDA <= 1 and 0 <= r3_CDA <= 1:
#                 interpolated_value_CDA = r1_CDA * f(C[0], C[1]) + r2_CDA * f(D[0], D[1]) + r3_CDA * f(A[0], A[1])
#                 return interpolated_value_CDA, A, B, C, D
#             else:
#                 print("Point y is not inside triangles ABC or CDA.")
#                 return None, None, None, None, None

#         else:
#             print("Error finding nearest points A, B, C, D.")
#             return None, None, None, None, None

    

#QUESTION 3 FINAL:


import numpy as np
import matplotlib.pyplot as plt

class InterpolationSolver:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.A = None
        self.B = None
        self.C = None
        self.D = None
    
    def find_nearest_point(self, quadrant):
        """
        Find the nearest point in X relative to y based on the specified quadrant.
        Quadrant numbering:
        1: x1 > y1, x2 > y2
        2: x1 > y1, x2 < y2
        3: x1 < y1, x2 < y2
        4: x1 < y1, x2 > y2
        """
        x1, x2 = self.X[:, 0], self.X[:, 1]
        y1, y2 = self.y[0], self.y[1]
        
        if quadrant == 1:
            mask = (x1 > y1) & (x2 > y2)
        elif quadrant == 2:
            mask = (x1 > y1) & (x2 < y2)
        elif quadrant == 3:
            mask = (x1 < y1) & (x2 < y2)
        elif quadrant == 4:
            mask = (x1 < y1) & (x2 > y2)
        else:
            raise ValueError("Quadrant must be between 1 and 4.")
        
        # Filter points in the specified quadrant
        X_quadrant = self.X[mask]
        
        if len(X_quadrant) == 0:
            return None
        
        # Find the nearest point in the quadrant
        distances = np.sqrt((X_quadrant[:, 0] - y1)**2 + (X_quadrant[:, 1] - y2)**2)
        nearest_index = np.argmin(distances)
        
        return X_quadrant[nearest_index]
    
    def compute_barycentric_coordinates(self, A, B, C):
        """
        Compute the barycentric coordinates of point y with respect to triangle ABC.
        """
        A1, A2 = A[0], A[1]
        B1, B2 = B[0], B[1]
        C1, C2 = C[0], C[1]
        y1, y2 = self.y[0], self.y[1]
        
        denominator = (B2 - C2) * (A1 - C1) + (C1 - B1) * (A2 - C2)
        
        r1 = ((B2 - C2) * (y1 - C1) + (C1 - B1) * (y2 - C2)) / denominator
        r2 = ((C2 - A2) * (y1 - C1) + (A1 - C1) * (y2 - C2)) / denominator
        r3 = 1 - r1 - r2
        
        return r1, r2, r3
    
    def interpolate_value(self):
        """
        Interpolate the value of f(y) based on the nearest points A, B, C, D.
        """
        if self.A is None or self.B is None or self.C is None or self.D is None:
            return np.nan
        
        r_ABC = self.compute_barycentric_coordinates(self.A, self.B, self.C)
        r_CDA = self.compute_barycentric_coordinates(self.C, self.D, self.A)
        
        if all(0 <= r <= 1 for r in r_ABC):
            interpolated_value = r_ABC[0] * self.f(self.A) + r_ABC[1] * self.f(self.B) + r_ABC[2] * self.f(self.C)
            return interpolated_value
        elif all(0 <= r <= 1 for r in r_CDA):
            interpolated_value = r_CDA[0] * self.f(self.C) + r_CDA[1] * self.f(self.D) + r_CDA[2] * self.f(self.A)
            return interpolated_value
        else:
            return np.nan
    
    # Dummy function f(point) - replace this with your actual function
    def f(self, point):
        return np.sum(point)
    
    def solve(self):
        # Find A, B, C, D
        self.A = self.find_nearest_point(1)
        self.B = self.find_nearest_point(2)
        self.C = self.find_nearest_point(3)
        self.D = self.find_nearest_point(4)
        
        if self.A is None or self.B is None or self.C is None or self.D is None:
            print("Cannot form the required quadrilateral. Return NaN.")
            return
        
        # Print values of A, B, C, D
        print(f"A = {self.A}")
        print(f"B = {self.B}")
        print(f"C = {self.C}")
        print(f"D = {self.D}")
        
        # Interpolate f(y)
        interpolated_value = self.interpolate_value()
        print(f"The interpolated value of f(y) at y = {self.y} is: {interpolated_value}")
        
        # Plot points X and y, and triangles ABC, CDA
        self.plot()

    def plot(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X[:, 0], self.X[:, 1], label='X points')
        plt.scatter(self.y[0], self.y[1], color='red', label='y')

        # Plot triangles ABC and CDA
        if self.A is not None and self.B is not None and self.C is not None:
            plt.plot([self.A[0], self.B[0], self.C[0], self.A[0]], [self.A[1], self.B[1], self.C[1], self.A[1]], 'b-', label='ABC')
            if self.D is not None:
                plt.plot([self.C[0], self.D[0], self.A[0], self.C[0]], [self.C[1], self.D[1], self.A[1], self.C[1]], 'g-', label='CDA')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Points X, y, and triangles ABC, CDA')
        plt.legend()
        plt.grid(True)
        plt.show()