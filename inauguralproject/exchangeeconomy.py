from types import SimpleNamespace

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

    def utility_A(self,x1A,x2A):
        par = self.par
        return x1A**par.alpha * x2A**(1 - par.alpha)
    
    def negative_utility_A(self, p1):
        x1A, x2A = self.demand_A(p1)
        return -self.utility_A(x1A, x2A)

    def utility_B(self,x1B,x2B):
        par = self.par
        return x1B**par.beta * x2B**(1 - par.beta)

    def demand_A(self,p1):
        w1A, w2A = self.par.w1A, self.par.w2A
        alpha = self.par.alpha
        x1A = alpha * (w1A * p1 + w2A) / p1
        x2A = (1 - alpha) * (w1A * p1 + w2A)
        return x1A, x2A

    def demand_B(self,p1):
        w1B, w2B = (1 - self.par.w1A), (1 - self.par.w2A)
        beta = self.par.beta
        x1B = beta * (w1B * p1 + w2B) / p1
        x2B = (1 - beta) * (w1B * p1 + w2B)
        return x1B, x2B

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        # Simplified, as we always know total endowment is 1 for each good
        eps1 = x1A + x1B - 1
        eps2 = x2A + x2B - 1

        return eps1,eps2
    
    #QUESTION 3 CODE:

    def check_market_clearing1(self,p1):
        x1A, _ = self.demand_A(p1)
        x1B, _ = self.demand_B(p1)

        # simplified as total endowment = 1 for each good
        epss1 = x1A + x1B - 1

        return epss1
    
    def check_market_clearing2(self,p1):
        _, x2A = self.demand_A(p1)
        _, x2B = self.demand_B(p1)
        
        # simplified as total endowment = 1 for each good
        epss2 = x2A + x2B - 1

        return epss2
      
    def find_equilibrium(self, p1_guess, kappa, eps, maxiter):
        import numpy as np
        p1 = p1_guess
        t = 0
        self.p2_star = 1  # Set p2 as the numeraire with a fixed value of 1

        for t in range(maxiter):
            # Calculate excess demands for goods 1 and 2
            eps1 = self.check_market_clearing1(p1)
            eps2 = self.check_market_clearing2(p1)

            # Print current status to monitor the iteration process
            print(f'Iteration {t}: p1 = {p1:.8f}, excess demand eps1 = {eps1:.8f}, eps2 = {eps2:.8f}')

            # Check if market is cleared for both goods, taking the absolute values of excess demands
            if np.abs(eps1) < eps and np.abs(eps2) < eps:
                print(f'Equilibrium found at iteration {t}: p1 = {p1:.8f}')
                self.p1_star = p1
                self.Z1 = eps1  # Store the excess demand for good 1 at equilibrium
                self.Z2 = eps2  # Store the excess demand for good 2 at equilibrium
                return p1  # Return immediately upon finding a solution

            # Adjust p1 based on the excess demand for good 1
            p1_update = p1 + kappa * eps1  # Notice the plus sign, not minus
            # Ensure that p1 does not become negative or zero
            if p1_update <= 0:
                print("Adjustment made p1 non-positive, reducing kappa and trying again.")
                kappa *= 0.5  # Reduce kappa if p1_update is non-positive
                p1 = max(p1 * 0.5, 1e-4)  # Make sure p1 is positive and not too close to zero
                continue  # Skip the rest of this iteration

            # If the change in p1 is very small, we might be close to convergence
            if np.abs(p1_update - p1) < eps:
                print(f'Convergence reached at iteration {t}: p1 = {p1_update:.8f}')
                self.p1_star = p1_update
                self.Z1 = eps1  # Store the excess demand for good 1 at convergence
                self.Z2 = eps2  # Store the excess demand for good 2 at convergence
                return p1_update  # Return if we've nearly converged

            p1 = p1_update  # Update p1 for the next iteration
            # Store the excess demand values for the current iteration
            self.Z1, self.Z2 = self.check_market_clearing(p1)

        # If we exit the loop without returning, no equilibrium was found
        print('Max iterations reached without finding equilibrium. Consider adjusting your parameters or initial guess.')
        self.p1_star = None  # Indicate that we did not find a solution
    
    def print_solution(self):
        # Check if the market-clearing price for p1 was found
        if hasattr(self, 'p1_star') and hasattr(self, 'p2_star'):
            text = 'Market-clearing prices:\n'
            text += f'p1 = {self.p1_star:.8f}\n'  # Adjusted to 8 decimal places
            text += f'p2 = {self.p2_star:.8f}\n\n' # p2 is the numeraire and set to 1, but formatted for consistency
        else:
            text = 'Market equilibrium price for p1 not found.\n'

        # Check if the excess demands were calculated
        if hasattr(self, 'Z1') and hasattr(self, 'Z2'):
            text += 'Corresponding excess demands at equilibrium:\n'
            text += f'Z1 (Good 1) = {self.Z1:+.8f}\n'  # Using the + sign to indicate the direction of excess demand and 8 decimal places
            text += f'Z2 (Good 2) = {self.Z2:+.8f}\n'  # Using the + sign to indicate the direction of excess demand and 8 decimal places
        else:
            text += 'Excess demands not calculated.\n'

        print(text)


    #QUESTION 6A:
    
    def utility_aggregate(self,x1A,x2A):
        par = self.par
        return x1A**par.alpha * x2A**(1 - par.alpha) + (1-x1A)**par.beta * (1-x2A)**(1 - par.beta)

    #QUESTION 8:
    
    def utility_aggregate_equilibrium(self,x):
        x1A = x[0]
        x2A = x[1]
        return -(self.utility_A(x1A, x2A) + self.utility_B(1 - x1A, 1 - x2A))

    def utility_A_mark(self,x1A,x2A):
        return x1A**(self.par.alpha) * x2A**(1 - self.par.alpha)

    def utility_B_mark(self,x1B,x2B):
        return x1B**(self.par.beta) * x2B**(1 - self.par.beta)