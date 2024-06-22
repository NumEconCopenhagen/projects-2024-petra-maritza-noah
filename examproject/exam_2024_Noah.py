#Q1.1
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

    def optimal_labor_supply(self, p):
        return (p * self.A * self.gamma / self.w) ** (1 / (1 - self.gamma))

    def optimal_output(self, labor):
        return self.A * labor ** self.gamma

    def implied_profits(self, p):
        labor = self.optimal_labor_supply(p)
        return (1 - self.gamma) / self.gamma * self.w * labor

    def check_market_clearing(self):
        p1_values = np.linspace(0.1, 2.0, 10)
        p2_values = np.linspace(0.1, 2.0, 10)
        results = []

        for p1 in p1_values:
            for p2 in p2_values:
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






