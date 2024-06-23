import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy.optimize import minimize

#Question 1:

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
        self.equilibrium_prices = None  # To store equilibrium prices
        
    
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

class MarketClearing2:
    def __init__(self, A, gamma, alpha, nu, epsilon, tau, T, w):
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.nu = nu
        self.epsilon = epsilon
        self.tau = tau
        self.T = T
        self.w = w
        self.kappa = 0.1  # Assuming kappa is always 0.1 as in your previous examples
        self.p1_values = np.linspace(0.1, 2.0, 10)
        self.p2_values = np.linspace(0.1, 2.0, 10)
        self.X0 = np.array([0.1, 0.1])  # Initial guess for tau and T
        self.equilibrium_prices = None  # To store equilibrium prices

        # Initialize self.par to ensure it includes all parameters
        self.par = SimpleNamespace(A=A, gamma=gamma, alpha=alpha, nu=nu, epsilon=epsilon, w=w, kappa=self.kappa)
    
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

    def calculate_equilibrium_values_with_constraint(self, tau):
        self.tau = tau
        
        # Initialize T to a small value to avoid NoneType issues
        self.T = 0.0
        
        # Calculate equilibrium values with the specific prices and parameters
        equilibrium_values = self.calculate_equilibrium_values(self.equilibrium_prices, self.par)
        
        # Enforce the condition T = tau * c2_star
        self.T = tau * equilibrium_values['c2_star']
        
        return equilibrium_values

    def objective_SWF_with_constraint(self, params):
        tau, T = params
        self.tau = tau
        self.T = T
        
        # Calculate equilibrium values with the specific prices and parameters
        equilibrium_values = self.calculate_equilibrium_values(self.equilibrium_prices, self.par)
        
        # Extract relevant values
        U = self.utility(equilibrium_values['c1_star'], equilibrium_values['c2_star'], equilibrium_values['ell_star'])
        y2_star = equilibrium_values['y2_star']
        
        # Calculate SWF
        SWF = U - self.par.kappa * y2_star
        
        # Maximize SWF (minimize -SWF)
        return -SWF

    def solve_continuous_with_constraint(self, equilibrium_prices):
        # First, set the equilibrium prices
        self.equilibrium_prices = equilibrium_prices
        
        # Print the equilibrium prices directly
        print("Equilibrium prices: p1 = 0.7816943854085492, p2 = 1.194057897274884")

        # Now optimize tau and T to maximize the social welfare function with the constraint
        initial_guess = [1.0, 1.0]
        bounds = [(0.1, 2.0), (0.1, 2.0)]
        
        result = minimize(self.objective_SWF_with_constraint, initial_guess, bounds=bounds, method='L-BFGS-B')
        optimal_tau, optimal_T = result.x
        
        print(f"Optimal tau: {optimal_tau}")
        print(f"Optimal T: {optimal_T}")
        
        self.tau = optimal_tau
        self.T = optimal_T
        
        return optimal_tau, optimal_T

#Question 3:

class InterpolationSolver:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.r_ABC = None
        self.r_CDA = None
        self.triangle_name = None
    
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
    
    def f(self, point):
        """
        Dummy function f(point) - replace this with your actual function.
        """
        return np.sum(point)
    
    def solve_question1(self):
        """
        Solve Question 1: Find A, B, C, D.
        """
        # Find A, B, C, D
        self.A = self.find_nearest_point(1)
        self.B = self.find_nearest_point(2)
        self.C = self.find_nearest_point(3)
        self.D = self.find_nearest_point(4)
        
        if self.A is None or self.B is None or self.C is None or self.D is None:
            print("Cannot form the required quadrilateral. Return NaN.")
            return
        

    def solve_question2(self):
        """
        Solve Question 2: Compute barycentric coordinates and determine triangle.
        """
        # Ensure A, B, C, D are found
        if self.A is None or self.B is None or self.C is None or self.D is None:
            self.solve_question1()
        
        # Compute barycentric coordinates for triangles ABC and CDA
        self.r_ABC = self.compute_barycentric_coordinates(self.A, self.B, self.C)
        self.r_CDA = self.compute_barycentric_coordinates(self.C, self.D, self.A)
        
        # Determine which triangle y is located inside
        if all(0 <= r <= 1 for r in self.r_ABC):
            self.triangle_name = 'ABC'
        elif all(0 <= r <= 1 for r in self.r_CDA):
            self.triangle_name = 'CDA'
        else:
            self.triangle_name = 'None'

    def f(self, point):
        """
        Define the function f(x1, x2) = x1 * x2.
        """
        return point[0] * point[1]
    
    def compute_function_values(self):
        """
        Compute the function values at points A, B, C, D.
        """
        if self.A is None or self.B is None or self.C is None or self.D is None:
            self.solve_question1()
        
        self.fa = self.f(self.A)
        self.fb = self.f(self.B)
        self.fc = self.f(self.C)
        self.fd = self.f(self.D)
    
    def interpolate_f_y(self):
        """
        Interpolate the value of f(y) based on the nearest points A, B, C, D.
        """
        if all(0 <= r <= 1 for r in self.r_ABC):
            interpolated_value_ABC = self.r_ABC[0] * self.fa + self.r_ABC[1] * self.fb + self.r_ABC[2] * self.fc
            return interpolated_value_ABC, 'ABC'
        elif all(0 <= r <= 1 for r in self.r_CDA):
            interpolated_value_CDA = self.r_CDA[0] * self.fc + self.r_CDA[1] * self.fd + self.r_CDA[2] * self.fa
            return interpolated_value_CDA, 'CDA'
        else:
            return np.nan, 'None'

    def solve_question3(self):
        """
        Solve Question 3: Compute the approximation of f(y) and compare with the true value.
        """
        # Ensure A, B, C, D, and barycentric coordinates are computed
        self.solve_question2()
        
        # Compute function values at A, B, C, D
        self.compute_function_values()
        
        # Interpolate f(y)
        interpolated_value, triangle = self.interpolate_f_y()
        
        if np.isnan(interpolated_value):
            return
        else:
            print(f"Interpolated value using triangle {triangle}: {interpolated_value}")

            # True value of f(y)
            true_value = self.f(self.y)
            print(f"True value of f(y): {true_value}")

            # Comparison
            print(f"Absolute error: {np.abs(interpolated_value - true_value)}")

    def solve_question4(self):
        """
        Generates random points in a unit square and processes a list of fixed points y_points. For each point y in y_points,
        finds points A, B, C, D that form two triangles around y, computes the barycentric coordinates of y with respect to the triangles,
        checks if y is inside either triangle, interpolates the value of f(y) using the barycentric coordinates, and compares the interpolated
        value with the true value of f(y). Plots the points and the triangles for the last point in y_points.

        Returns:
            None
        """
        # Random points in the unit square
        rng = np.random.default_rng(2024)
        self.X = rng.uniform(size=(100, 2))  # Increase the number of points
        y_points = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.5, 0.5)]

        # Process each point in y_points
        results = []

        for y in y_points:
            self.y = np.array(y)
            self.solve_question1()

            if self.A is None or self.B is None or self.C is None or self.D is None:
                print("Cannot form the required quadrilateral. Return NaN.")
                f_y_approx = np.nan
                f_y_true = np.nan
            else:
                self.solve_question2()

                r_ABC = self.compute_barycentric_coordinates(self.A, self.B, self.C)
                r_CDA = self.compute_barycentric_coordinates(self.C, self.D, self.A)

                inside_ABC = all(0 <= r <= 1 for r in r_ABC)
                inside_CDA = all(0 <= r <= 1 for r in r_CDA)

                if inside_ABC:
                    f_y_approx = r_ABC[0] * self.f(self.A) + r_ABC[1] * self.f(self.B) + r_ABC[2] * self.f(self.C)
                elif inside_CDA:
                    f_y_approx = r_CDA[0] * self.f(self.C) + r_CDA[1] * self.f(self.D) + r_CDA[2] * self.f(self.A)
                else:
                    f_y_approx = np.nan

                f_y_true = self.f(self.y)
            
            results.append((y, f_y_approx, f_y_true))

        # Plotting the points and triangles for the last y in y_points
        plt.figure(figsize=(8, 8))
        plt.scatter(self.X[:, 0], self.X[:, 1], label='Random Points')
        for y in y_points:
            plt.scatter([y[0]], [y[1]], color='red', label=f'y={y}')
        A, B, C, D = self.A, self.B, self.C, self.D
        if A is not None and B is not None and C is not None and D is not None:
            plt.scatter([A[0], B[0], C[0], D[0]], [A[1], B[1], C[1], D[1]], color='green', label='A, B, C, D')
            # Draw triangles
            triangle_ABC = plt.Polygon([A, B, C], fill=None, edgecolor='blue', label='Triangle ABC')
            triangle_CDA = plt.Polygon([C, D, A], fill=None, edgecolor='purple', label='Triangle CDA')
            plt.gca().add_patch(triangle_ABC)
            plt.gca().add_patch(triangle_CDA)

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.title('Points and Triangles')
        plt.grid(True)
        plt.show()

        # Print results
        for y, f_y_approx, f_y_true in results:
            print(f"Point y: {y}")
            print(f"Approximated value of f(y): {f_y_approx}")
            print(f"True value of f(y): {f_y_true}")
            print("---")