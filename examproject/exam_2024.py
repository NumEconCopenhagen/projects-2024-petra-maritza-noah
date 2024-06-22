#Question 3

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

