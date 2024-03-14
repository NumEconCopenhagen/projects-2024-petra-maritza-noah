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
      
    def find_equilibrium(self, p1_guess, eps, kappa, maxiter):
        import numpy as np
        p2 = 1  # p2 is the numeraire and is set to 1
        p1 = p1_guess
        t = 0

        while True:
            # Calculate excess demands
            eps1 = self.check_market_clearing1(p1)
            eps2 = self.check_market_clearing2(p1)

            # Print progress
            if t < 5 or t % 25 == 0:
                print(f'{t:3d}: p1 = {p1:12.8f}, eps1 = {eps1:14.8f}, eps2 = {eps2:14.8f}')
            elif t == 5:
                print('   ...')

            # Check if market is cleared for good 1
            if np.abs(eps1) < eps:
                break

            # Update p1 based on excess demand for good 1
            p1 = p1 - kappa * eps1

            t += 1
            if t >= maxiter:
                print('Max iterations reached without finding equilibrium.')
                break

        if np.abs(eps1) < eps:
            self.p1_star = p1
            self.p2_star = p2  # p2 is always 1
            print(f'Equilibrium found: p1 = {self.p1_star}, p2 = {self.p2_star}')
        else:
            print('Equilibrium not found.')