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
        p1_last = p1  # To track the previous iteration's p1 for convergence check
    
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
                return p1  # Return immediately upon finding a solution

        # Adjust p1 based on the excess demand for good 1
        # Ensure that p1 does not become negative or zero
            p1_update = p1 - kappa * eps1
            if p1_update <= 0:
                kappa *= 0.5  # Reduce kappa if p1_update is non-positive
                continue  # Skip the rest of this iteration

            p1 = p1_update  # Update p1 if p1_update is positive
        
            # If the change in p1 is very small, we might be close to convergence
            if np.abs(p1 - p1_last) < eps:
                print(f'Price convergence reached at iteration {t}: p1 = {p1:.8f}')
                self.p1_star = p1
                return p1  # Return if we've nearly converged

            p1_last = p1  # Update p1_last for the next iteration

    # If we exit the loop without returning, no equilibrium was found
        print('Max iterations reached without finding equilibrium. Consider adjusting your parameters or initial guess.')
        self.p1_star = None  # Indicate that we did not find a solution