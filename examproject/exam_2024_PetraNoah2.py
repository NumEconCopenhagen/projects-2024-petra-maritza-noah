#Question 3:

import numpy as np
import matplotlib as plt

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


    def solve_question4(self, Y):
        """
        Solve Question 4: Compute the approximation of f(y) for each point in Y and compare with the true value.
        """
        for point in Y:
            self.y = point
            self.solve_question3()
            print()  # Print empty line for separation between results
