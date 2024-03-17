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

    def demand_A(self,p1, p2):
        par = self.par
        x1A = par.alpha * (p1 * par.w1A + p2 * par.w2A) / p1
        x2A = (1 - par.alpha) * (p1 * par.w1A + p2 * par.w2A) / p2
        return x1A, x2A

    def demand_B(self,p1, p2):
        par = self.par
        x1B = par.beta * (p1 * par.w1A + p2 * par.w2A) / p1
        x2B = (1 - par.beta) * (p1 * par.w1A + p2 * par.w2A) / p2
        return x1B, x2B

    #Check Walras market equilibrium:
    
    def check_market_clearing(self,p1,p2):

        par = self.par

        x1A,x2A = self.demand_A(p1, p2)
        x1B,x2B = self.demand_B(p1, p2)

        eps1 = x1A + x1B - 1
        eps2 = x2A + x2B - 1

        return eps1,eps2
    
    #QUESTION 3 CODE:

    def check_market_clearing1(self,p1, p2):

        x1A, _ = self.demand_A(p1, p2)
        x1B, _ = self.demand_B(p1, p2)

        epss1 = x1A + x1B - 1

        return epss1
        #epss1 is excess demand for good1
    
    def check_market_clearing2(self,p1, p2):

        _,x2A = self.demand_A(p1, p2)
        _,x2B = self.demand_B(p1, p2)

        epss2 = x2A + x2B - 1

        return epss2
        #epss2 is excess demand for good2
      
    def find_equilibrium(self,p1_guess,p2, N, k, eps, kappa, maxiter):
        import numpy as np
        t = 0
        p1 = p1_guess
        epss1 = self.check_market_clearing1(p1, p2)
        
        # using a while loop as we don't know number of iterations a priori
        while True:

            # a. step 1: excess demand
            #epss1 = self.check_market_clearing1(p1, p2)
            Z1 = epss1

            # b: step 2: stop?
            if  np.abs(Z1) < eps or t >= maxiter:
                print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand -> {Z1:14.8f}')
                break    
            
            # c. Print the first 5 and every 25th iteration using the modulus operator 
            if t < 5 or t%25 == 0:
                print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand -> {Z1:14.8f}')
            elif t == 5:
                print('   ...')
            
            # d. step 3: update p1
            p1 = p1 + kappa*Z1/N

            #recalculate epss1 using the updated p1
            epss1 = self.check_market_clearing1(p1,p2)
            
            # e. step 4: update counter and return to step 1
            t += 1    


        # Check if solution is found 
        if np.abs(Z1) < eps:
            # Store equilibrium prices
            self.p1_star = p1 
            self.p2_star = p2

            # Store equilibrium excess demand 
            self.Z1 = Z1
            self.Z2 = self.check_market_clearing2(self.p1_star, self.p2_star)

            # Make sure that Walras' law is satisfied
            if not np.abs(self.Z2)< eps:
                print('The market for good 2 was not cleared')
                print(f'Z2 = {self.Z2}')

        else:
            print('Solution was not found')


    #Display the solution
    def print_solution(self):

        text = 'Solution to market equilibrium:\n'
        text += f'p1 = {self.p1_star:5.3f}\np2 = {self.p2_star:5.3f}\n\n'

        text += 'Excess demands are:\n'
        text += f'Z1 = {self.Z1}\n'
        text += f'Z2 = {self.Z2}'
        print(text)


#QUESTION 4A AND 4B: