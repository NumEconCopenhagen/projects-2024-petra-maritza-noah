
# import numpy as np
# import matplotlib.pyplot as plt

# class InterpolationSolver:
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y
#         self.A = None
#         self.B = None
#         self.C = None
#         self.D = None
#         self.r_ABC = None
#         self.r_CDA = None
#         self.triangle_name = None
    
#     def find_nearest_point(self, quadrant):
#         """
#         Find the nearest point in X relative to y based on the specified quadrant.
#         Quadrant numbering:
#         1: x1 > y1, x2 > y2
#         2: x1 > y1, x2 < y2
#         3: x1 < y1, x2 < y2
#         4: x1 < y1, x2 > y2
#         """
#         x1, x2 = self.X[:, 0], self.X[:, 1]
#         y1, y2 = self.y[0], self.y[1]
        
#         if quadrant == 1:
#             mask = (x1 > y1) & (x2 > y2)
#         elif quadrant == 2:
#             mask = (x1 > y1) & (x2 < y2)
#         elif quadrant == 3:
#             mask = (x1 < y1) & (x2 < y2)
#         elif quadrant == 4:
#             mask = (x1 < y1) & (x2 > y2)
#         else:
#             raise ValueError("Quadrant must be between 1 and 4.")
        
#         # Filter points in the specified quadrant
#         X_quadrant = self.X[mask]
        
#         if len(X_quadrant) == 0:
#             return None
        
#         # Find the nearest point in the quadrant
#         distances = np.sqrt((X_quadrant[:, 0] - y1)**2 + (X_quadrant[:, 1] - y2)**2)
#         nearest_index = np.argmin(distances)
        
#         return X_quadrant[nearest_index]
    
#     def compute_barycentric_coordinates(self, A, B, C):
#         """
#         Compute the barycentric coordinates of point y with respect to triangle ABC.
#         """
#         A1, A2 = A[0], A[1]
#         B1, B2 = B[0], B[1]
#         C1, C2 = C[0], C[1]
#         y1, y2 = self.y[0], self.y[1]
        
#         denominator = (B2 - C2) * (A1 - C1) + (C1 - B1) * (A2 - C2)
        
#         r1 = ((B2 - C2) * (y1 - C1) + (C1 - B1) * (y2 - C2)) / denominator
#         r2 = ((C2 - A2) * (y1 - C1) + (A1 - C1) * (y2 - C2)) / denominator
#         r3 = 1 - r1 - r2
        
#         return r1, r2, r3
    
#     def interpolate_value(self):
#         """
#         Interpolate the value of f(y) based on the nearest points A, B, C, D.
#         """
#         if self.A is None or self.B is None or self.C is None or self.D is None:
#             return np.nan
        
#         r_ABC = self.compute_barycentric_coordinates(self.A, self.B, self.C)
#         r_CDA = self.compute_barycentric_coordinates(self.C, self.D, self.A)
        
#         if all(0 <= r <= 1 for r in r_ABC):
#             interpolated_value = r_ABC[0] * self.f(self.A) + r_ABC[1] * self.f(self.B) + r_ABC[2] * self.f(self.C)
#             return interpolated_value
#         elif all(0 <= r <= 1 for r in r_CDA):
#             interpolated_value = r_CDA[0] * self.f(self.C) + r_CDA[1] * self.f(self.D) + r_CDA[2] * self.f(self.A)
#             return interpolated_value
#         else:
#             return np.nan
    
#     def f(self, point):
#         """
#         Dummy function f(point) - replace this with your actual function.
#         """
#         return np.sum(point)
    
#     def compute_function_values(self):
#         """
#         Compute the function values at points A, B, C, D.
#         """
#         self.fa = self.f(self.A)
#         self.fb = self.f(self.B)
#         self.fc = self.f(self.C)
#         self.fd = self.f(self.D)

#     def interpolate_f_y(self):
#         """
#         Interpolate the value of f(y) based on the nearest points A, B, C, D.
#         """
#         if all(0 <= r <= 1 for r in self.r_ABC):
#             interpolated_value_ABC = self.r_ABC[0] * self.fa + self.r_ABC[1] * self.fb + self.r_ABC[2] * self.fc
#             return interpolated_value_ABC, 'ABC'
#         elif all(0 <= r <= 1 for r in self.r_CDA):
#             interpolated_value_CDA = self.r_CDA[0] * self.fc + self.r_CDA[1] * self.fd + self.r_CDA[2] * self.fa
#             return interpolated_value_CDA, 'CDA'
#         else:
#             return np.nan, 'None'
    
#     def solve_question1(self):
#         """
#         Solve Question 1: Find A, B, C, D and plot.
#         """
#         # Find A, B, C, D
#         self.A = self.find_nearest_point(1)
#         self.B = self.find_nearest_point(2)
#         self.C = self.find_nearest_point(3)
#         self.D = self.find_nearest_point(4)
        
#         if self.A is None or self.B is None or self.C is None or self.D is None:
#             print("Cannot form the required quadrilateral. Return NaN.")
#             return
        
#         # Plot points X and y, and triangles ABC, CDA
#         self.plot()

#         # Print values of A, B, C, D
#         print(f"A = {self.A}")
#         print(f"B = {self.B}")
#         print(f"C = {self.C}")
#         print(f"D = {self.D}")
    
#     def solve_question2(self):
#         """
#         Solve Question 2: Compute barycentric coordinates and determine triangle.
#         """
#         #Find A, B, C, D
#         self.A = self.find_nearest_point(1)
#         self.B = self.find_nearest_point(2)
#         self.C = self.find_nearest_point(3)
#         self.D = self.find_nearest_point(4)
        
#         if self.A is None or self.B is None or self.C is None or self.D is None:
#             print("Cannot form the required quadrilateral. Return NaN.")
#             return
        
#         # Compute barycentric coordinates for triangles ABC and CDA
#         self.r_ABC = self.compute_barycentric_coordinates(self.A, self.B, self.C)
#         self.r_CDA = self.compute_barycentric_coordinates(self.C, self.D, self.A)
        
#         # Determine which triangle y is located inside
#         if all(0 <= r <= 1 for r in self.r_ABC):
#             self.triangle_name = 'ABC'
#         elif all(0 <= r <= 1 for r in self.r_CDA):
#             self.triangle_name = 'CDA'
#         else:
#             self.triangle_name = 'None'
        
#         # Print values of A, B, C, D
#         # print(f"A = {self.A}")
#         # print(f"B = {self.B}")
#         # print(f"C = {self.C}")
#         # print(f"D = {self.D}")
        
#         # Print barycentric coordinates and triangle location
#         print(f"Barycentric coordinates for y: ABC{self.r_ABC}, CDA{self.r_CDA}")
#         print(f"Point y is located inside triangle {self.triangle_name}.")
        
#         # Interpolate f(y)
#         interpolated_value = self.interpolate_value()
#         print(f"The interpolated value of f(y) at y = {self.y} is: {interpolated_value}")
        
#         # Plot points X and y, and triangles ABC, CDA
#         #self.plot()


#     def solve_question3(self):
#         """
#         Solve Question 3: Compute the approximation of f(y) and compare with the true value.
#         """
#         # Ensure A, B, C, D, and barycentric coordinates are computed
#         self.solve_question2()
        
#         # Compute function values at A, B, C, D
#         self.compute_function_values()
        
#         # Interpolate f(y)
#         interpolated_value, triangle = self.interpolate_f_y()
        
#         if np.isnan(interpolated_value):
#             return
#         else:
#             print(f"Interpolated value using triangle {triangle}: {interpolated_value}")

#             # True value of f(y)
#             true_value = self.f(self.y)
#             print(f"True value of f(y): {true_value}")

#             # Comparison
#             print(f"Absolute error: {np.abs(interpolated_value - true_value)}")


#     def f1(x1, x2):
#         return x1 * x2


#     def find_nearest_point2(self, X, y):
#         """
#         Find the nearest point in X relative to y.
#         """
#         distances = np.sqrt(np.sum((X - y)**2, axis=1))
#         nearest_index = np.argmin(distances)
        
#         return X[nearest_index]
    
#     # def solve_question4(self, Y):
#     #     """
#     #     Solve Question 4: Compute the approximation of f(y) and compare with the true value for all points in Y.
#     #     """
#     #     rng = np.random.default_rng(2024)
#     #     X = rng.uniform(size=(50, 2))
#     #     Y = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.8, 0.2), (0.5, 0.5)]

#     #     # for y in Y:
#     #     #     # Find nearest points A, B, C, D
#     #     #     try:
#     #     #         A, B, C, D = self.find_nearest_point(X, y)
#     #     #     except IndexError:
#     #     #         print(f"Not enough points in X to form quadrilateral ABCD for y={y}.")
#     #     #         continue

#     #     for y in Y:
#     #         # Find nearest points A, B, C, D using find_nearest_point2
#     #         try:
#     #             A = self.find_nearest_point2(self.X, y)
#     #             B = self.find_nearest_point2(self.X, y)
#     #             C = self.find_nearest_point2(self.X, y)
#     #             D = self.find_nearest_point2(self.X, y)
#     #         except IndexError:
#     #             print(f"Not enough points in X to form quadrilateral ABCD for y={y}.")
#     #             continue

#     #         #Compute barycentric coordinates for triangles ABC and CDA
#     #         if A is not None and B is not None and C is not None:
#     #             r1_ABC, r2_ABC, r3_ABC = self.compute_barycentric_coordinates(y, A, B, C)
#     #             r1_CDA, r2_CDA, r3_CDA = self.compute_barycentric_coordinates(y, C, D, A)

            
#     #             # Define the function values at A, B, C, D
#     #             fa = self.f1(A[0], A[1])
#     #             fb = self.f1(B[0], B[1])
#     #             fc = self.f1(C[0], C[1])
#     #             fd = self.f1(D[0], D[1])

#     #             # Interpolate f(y)
#     #             if 0 <= r1_ABC <= 1 and 0 <= r2_ABC <= 1 and 0 <= r3_ABC <= 1:
#     #                 interpolated_value_ABC = r1_ABC * fa + r2_ABC * fb + r3_ABC * fc
#     #                 print(f"For y={y}: Interpolated value using triangle ABC: {interpolated_value_ABC}")
#     #             elif 0 <= r1_CDA <= 1 and 0 <= r2_CDA <= 1 and 0 <= r3_CDA <= 1:
#     #                 interpolated_value_CDA = r1_CDA * fc + r2_CDA * fd + r3_CDA * fa
#     #                 print(f"For y={y}: Interpolated value using triangle CDA: {interpolated_value_CDA}")
#     #             else:
#     #                 print(f"For y={y}: Point y is not inside triangles ABC or CDA.")

#     #             # True value of f(y)
#     #             true_value = self.f1(y[0], y[1])
#     #             print(f"For y={y}: True value of f(y): {true_value}")

#     #             # Comparison
#     #             print(f"For y={y}: Absolute error: {np.abs(interpolated_value_ABC - true_value)}\n")

#     #         else:
#     #             print(f"Error finding nearest points A, B, C, D for y={y}.")

#     def compute_barycentric_coordinates_for_question4(self, A, B, C):
#         """
#         Compute barycentric coordinates for triangles ABC and CDA specifically for solve_question4.
#         """
#         A1, A2 = A[0], A[1]
#         B1, B2 = B[0], B[1]
#         C1, C2 = C[0], C[1]
#         y1, y2 = self.y[0], self.y[1]

#         denominator = (B2 - C2) * (A1 - C1) + (C1 - B1) * (A2 - C2)
#         r1 = ((y2 - C2) * (A1 - C1) + (C1 - y1) * (A2 - C2)) / denominator
#         r2 = ((y2 - A2) * (B1 - A1) + (A1 - y1) * (B2 - A2)) / denominator
#         r3 = 1 - r1 - r2

#         return r1, r2, r3

#     def solve_question4(self, Y):
#         """
#         Solve Question 4: Compute the approximation of f(y) and compare with the true value for all points in Y.
#         """
#         for y in Y:
#             self.y = y  # Set self.y to the current point y
            
#             try:
#                 A = self.find_nearest_point2(self.X, self.y)
#                 B = self.find_nearest_point2(self.X, self.y)
#                 C = self.find_nearest_point2(self.X, self.y)
#                 D = self.find_nearest_point2(self.X, self.y)
#             except IndexError:
#                 print(f"Not enough points in X to form quadrilateral ABCD for y={self.y}.")
#                 continue

#             if A is None or B is None or C is None or D is None:
#                 print(f"Error finding nearest points A, B, C, D for y={self.y}.")
#                 continue

#             # Extract components of A, B, C, D
#             A1, A2 = A[0], A[1]
#             B1, B2 = B[0], B[1]
#             C1, C2 = C[0], C[1]
#             D1, D2 = D[0], D[1]

#             # Compute barycentric coordinates for triangles ABC and CDA
#             try:
#                 r1_ABC, r2_ABC, r3_ABC = self.compute_barycentric_coordinates_for_question4(A, B, C)
#                 r1_CDA, r2_CDA, r3_CDA = self.compute_barycentric_coordinates_for_question4(C, D, A)
#             except TypeError as e:
#                 print(f"Error computing barycentric coordinates: {e}")
#                 continue
            
        
#             #Define the function values at A, B, C, D
#             fa = self.f1(A1, A2)
#             fb = self.f1(B1, B2)
#             fc = self.f1(C1, C2)
#             fd = self.f1(D1, D2)

            
#             # Interpolate f(y) using ABC triangle
#             if 0 <= r1_ABC <= 1 and 0 <= r2_ABC <= 1 and 0 <= r3_ABC <= 1:
#                 interpolated_value_ABC = r1_ABC * fa + r2_ABC * fb + r3_ABC * fc
#                 print(f"For y={self.y}: Interpolated value using triangle ABC: {interpolated_value_ABC}")
#             else:
#                 print(f"For y={self.y}: Point y is not inside triangle ABC.")

#             # Interpolate f(y) using CDA triangle
#             if 0 <= r1_CDA <= 1 and 0 <= r2_CDA <= 1 and 0 <= r3_CDA <= 1:
#                 interpolated_value_CDA = r1_CDA * fc + r2_CDA * fd + r3_CDA * fa
#                 print(f"For y={self.y}: Interpolated value using triangle CDA: {interpolated_value_CDA}")
#             else:
#                 print(f"For y={self.y}: Point y is not inside triangle CDA.")

#             # True value of f(y)
#             true_value = self.f1(self.y[0], self.y[1])
#             print(f"For y={self.y}: True value of f(y): {true_value}")

#             # Comparison
#             if 0 <= r1_ABC <= 1 and 0 <= r2_ABC <= 1 and 0 <= r3_ABC <= 1:
#                 absolute_error_ABC = np.abs(interpolated_value_ABC - true_value)
#                 print(f"For y={self.y}: Absolute error using triangle ABC: {absolute_error_ABC}")

#             if 0 <= r1_CDA <= 1 and 0 <= r2_CDA <= 1 and 0 <= r3_CDA <= 1:
#                 absolute_error_CDA = np.abs(interpolated_value_CDA - true_value)
#                 print(f"For y={self.y}: Absolute error using triangle CDA: {absolute_error_CDA}")

#             print()  # For separating outputs


    
            
#     def plot(self):
#         plt.figure(figsize=(8, 6))
#         plt.scatter(self.X[:, 0], self.X[:, 1], label='X points')
#         plt.scatter(self.y[0], self.y[1], color='red', label='y')

#         # Plot triangles ABC and CDA
#         if self.A is not None and self.B is not None and self.C is not None:
#             plt.plot([self.A[0], self.B[0], self.C[0], self.A[0]], [self.A[1], self.B[1], self.C[1], self.A[1]], 'b-', label='ABC')
#             if self.D is not None:
#                 plt.plot([self.C[0], self.D[0], self.A[0], self.C[0]], [self.C[1], self.D[1], self.A[1], self.C[1]], 'g-', label='CDA')

#         plt.xlabel('x1')
#         plt.ylabel('x2')
#         plt.title('Points X, y, and triangles ABC, CDA')
#         plt.legend()
#         plt.grid(True)
#         plt.show()


import numpy as np
import matplotlib.pyplot as plt

def f1(x1, x2):
    return x1 * x2

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

        X_quadrant = self.X[mask]

        if len(X_quadrant) == 0:
            return None

        distances = np.sqrt((X_quadrant[:, 0] - y1) ** 2 + (X_quadrant[:, 1] - y2) ** 2)
        nearest_index = np.argmin(distances)

        return X_quadrant[nearest_index]

    def compute_barycentric_coordinates(self, A, B, C):
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
        if self.A is None or self.B is None or self.C is None or self.D is None:
            return np.nan

        r_ABC = self.compute_barycentric_coordinates(self.A, self.B, self.C)
        r_CDA = self.compute_barycentric_coordinates(self.C, self.D, self.A)

        if all(0 <= r <= 1 for r in r_ABC):
            interpolated_value = (
                r_ABC[0] * f1(*self.A)
                + r_ABC[1] * f1(*self.B)
                + r_ABC[2] * f1(*self.C)
            )
            return interpolated_value
        elif all(0 <= r <= 1 for r in r_CDA):
            interpolated_value = (
                r_CDA[0] * f1(*self.C)
                + r_CDA[1] * f1(*self.D)
                + r_CDA[2] * f1(*self.A)
            )
            return interpolated_value
        else:
            return np.nan

    def solve_question1(self):
        self.A = self.find_nearest_point(1)
        self.B = self.find_nearest_point(2)
        self.C = self.find_nearest_point(3)
        self.D = self.find_nearest_point(4)

        if self.A is None or self.B is None or self.C is None or self.D is None:
            print("Cannot form the required quadrilateral. Return NaN.")
            return

        #self.plot()

        # print(f"A = {self.A}")
        # print(f"B = {self.B}")
        # print(f"C = {self.C}")
        # print(f"D = {self.D}")

    def solve_question2(self):
        self.A = self.find_nearest_point(1)
        self.B = self.find_nearest_point(2)
        self.C = self.find_nearest_point(3)
        self.D = self.find_nearest_point(4)

        if self.A is None or self.B is None or self.C is None or self.D is None:
            print("Cannot form the required quadrilateral. Return NaN.")
            return

        self.r_ABC = self.compute_barycentric_coordinates(self.A, self.B, self.C)
        self.r_CDA = self.compute_barycentric_coordinates(self.C, self.D, self.A)

        if all(0 <= r <= 1 for r in self.r_ABC):
            self.triangle_name = 'ABC'
        elif all(0 <= r <= 1 for r in self.r_CDA):
            self.triangle_name = 'CDA'
        else:
            self.triangle_name = 'None'

        print(f"Barycentric coordinates for y: ABC{self.r_ABC}, CDA{self.r_CDA}")
        print(f"Point y is located inside triangle {self.triangle_name}.")

        interpolated_value = self.interpolate_value()
        print(f"The interpolated value of f(y) at y = {self.y} is: {interpolated_value}")

        #self.plot()

    def solve_question3(self):
        self.solve_question2()

        if any(coordinate is None for coordinate in [self.A, self.B, self.C, self.D]):
            print("Cannot form the required quadrilateral. Return NaN.")
            return

        interpolated_value = self.interpolate_value()
        if np.isnan(interpolated_value):
            print("Cannot interpolate f(y). Return NaN.")
        else:
            print(f"Interpolated value: {interpolated_value}")

            true_value = f1(*self.y)
            print(f"True value of f(y): {true_value}")

    def solve_question4(self, Y):
        X_orig = self.X
        interpolated_values = []

        for y in Y:
            self.y = y
            self.solve_question2()
            interpolated_value = self.interpolate_value()
            interpolated_values.append(interpolated_value)
        
        return np.array(interpolated_values)

    # def plot(self):
    #     plt.scatter(self.X[:, 0], self.X[:, 1], color='blue', label='Points in X')
    #     plt.scatter(*self.y, color='red', label='Point y')
    #     plt.scatter(*self.A, color='green', label='Point A')
    #     plt.scatter(*self.B, color='green', label='Point B')
    #     plt.scatter(*self.C, color='green', label='Point C')
    #     plt.scatter(*self.D, color='green', label='Point D')
    #     plt.plot([self.A[0], self.B[0], self.C[0], self.A[0]], [self.A[1], self.B[1], self.C[1], self.A[1]], 'k-', label='Triangle ABC')
    #     plt.plot([self.C[0], self.D[0], self.A[0], self.C[0]], [self.C[1], self.D[1], self.A[1], self.C[1]], 'k-', label='Triangle CDA')
    #     plt.legend()
    #     plt.xlabel('x1')
    #     plt.ylabel('x2')
    #     plt.title('Interpolation Solver')
    #     plt.grid(True)
    #     plt.show()