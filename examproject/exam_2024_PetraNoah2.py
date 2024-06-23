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

    # def solve_question4(self, Y):
    #     results = []
    #     for y in Y:
    #         self.y = y
    #         print(f"Processing point: {y}")
    #         interpolated_value, triangle = self.solve_question3()
    #         true_value = self.f(y) if interpolated_value is not np.nan else np.nan
    #         results.append((y, interpolated_value, triangle, true_value))
        
    #     for y, interpolated_value, triangle, true_value in results:
    #         print(f"Point {y}:")
    #         print(f"  Interpolated value using triangle {triangle}: {interpolated_value}")
    #         print(f"  True value of f(y): {true_value}")
    #         if not np.isnan(interpolated_value):
    #             print(f"  Absolute error: {np.abs(interpolated_value - true_value)}")
    #         print()

    def question4():
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
        X = rng.uniform(size=(50, 2))
        y_points = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.5, 0.5)]

        # Define the function f
        f = lambda x: x[0] * x[1]
        F = np.array([f(x) for x in X])

        # Function to calculate distance
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        # Function to find the points A, B, C, D
        def find_points(y):
            candidates_A = [point for point in X if point[0] > y[0] and point[1] > y[1]]
            candidates_B = [point for point in X if point[0] > y[0] and point[1] < y[1]]
            candidates_C = [point for point in X if point[0] < y[0] and point[1] < y[1]]
            candidates_D = [point for point in X if point[0] < y[0] and point[1] > y[1]]

            A = min(candidates_A, key=lambda point: distance(point, y)) if candidates_A else None
            B = min(candidates_B, key=lambda point: distance(point, y)) if candidates_B else None
            C = min(candidates_C, key=lambda point: distance(point, y)) if candidates_C else None
            D = min(candidates_D, key=lambda point: distance(point, y)) if candidates_D else None

            return A, B, C, D

        # Function to compute barycentric coordinates
        def barycentric_coords(x, y, z, p):
            denom = (y[1] - z[1]) * (x[0] - z[0]) + (z[0] - y[0]) * (x[1] - z[1])
            a = ((y[1] - z[1]) * (p[0] - z[0]) + (z[0] - y[0]) * (p[1] - z[1])) / denom
            b = ((z[1] - x[1]) * (p[0] - z[0]) + (x[0] - z[0]) * (p[1] - z[1])) / denom
            c = 1 - a - b
            return a, b, c

        # Check if point y is inside the triangle using barycentric coordinates
        def is_inside_triangle(r):
            return all(0 <= coord <= 1 for coord in r)

        # Function to interpolate using barycentric coordinates
        def interpolate(f_vals, bary_coords):
            return sum(f_val * bary_coord for f_val, bary_coord in zip(f_vals, bary_coords))

        # Process each point in y_points
        results = []

        for y in y_points:
            A, B, C, D = find_points(y)

            if A is None or B is None or C is None or D is None:
                f_y_approx = np.nan
            else:
                r_ABC = barycentric_coords(A, B, C, y)
                r_CDA = barycentric_coords(C, D, A, y)

                inside_ABC = is_inside_triangle(r_ABC)
                inside_CDA = is_inside_triangle(r_CDA)

                if inside_ABC:
                    f_y_approx = interpolate([f(A), f(B), f(C)], r_ABC)
                elif inside_CDA:
                    f_y_approx = interpolate([f(C), f(D), f(A)], r_CDA)
                else:
                    f_y_approx = np.nan

            f_y_true = f(y)
            results.append((y, f_y_approx, f_y_true))

        # Plotting the points and triangles for the last y in y_points
        plt.figure(figsize=(8, 8))
        plt.scatter(X[:, 0], X[:, 1], label='Random Points')
        for y in y_points:
            plt.scatter([y[0]], [y[1]], color='red', label=f'y={y}')
        A, B, C, D = find_points(y_points[-1])
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

    
        

