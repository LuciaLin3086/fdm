import numpy as np


# input variable
S0 = 50
K = 50
r = 0.05
q = 0.01
sigma = 0.4
T = 0.5
Smax = 100
Smin = 0
m = 100
n = 1000

###############
## Version 2 ##
###############

class ExplicitFiniteDifference:

    def __init__(self, S0, K, r, q, sigma, T, Smax, Smin, m, n, C_P, E_A):
        self.S0 = S0
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma
        self.T = T
        self.Smax = Smax
        self.Smin = Smin
        self.m = m
        self.n = n
        self.C_P = C_P
        self.E_A = E_A

        self.delta_T = self.T / self.n
        self.delta_S = (self.Smax - self.Smin) / self.m


    def get_coefficient_matrix(self):
        #########################################
        ## 將a,b,c和coeff_matrix用同一個for迴圈一起算
        #########################################

        coeff_matrix = np.zeros((self.m - 1, self.m + 1))

        for j in range(1, self.m):  # j = 1,2,...,m-2,m-1
            a = (- (self.r - self.q) / 2 * j * self.delta_T + self.sigma ** 2 * j ** 2 * self.delta_T / 2) / (
                        1 + self.r * self.delta_T)
            b = (1 - self.sigma ** 2 * j ** 2 * self.delta_T) / (1 + self.r * self.delta_T)
            c = ((self.r - self.q) / 2 * j * self.delta_T + self.sigma ** 2 * j ** 2 * self.delta_T / 2) / (
                        1 + self.r * self.delta_T)

            # 注意計算a,b,c時的j是如講義中由下往上數的
            # coeff_matrix的index是從上往下數的，所以 row: m - j - 1
            coeff_matrix[self.m - j - 1, self.m - j - 1] = c # diagonal element
            coeff_matrix[self.m - j - 1, (self.m - j - 1) + 1] = b
            coeff_matrix[self.m - j - 1, (self.m - j - 1) + 2] = a

        return coeff_matrix


    def get_payoff(self, price):
        if self.C_P == "C": # call
            payoff = max(price - self.K, 0)
        elif self.C_P == "P": # put
            payoff = max(self.K - price, 0)

        return payoff

    def backward_induction(self):
        ###########################################
        ## 覆值時都是整個list一起，而不用對應每個element
        ###########################################

        f_matrix = np.zeros((self.m + 1, self.n + 1))

        # 先算maturity payoff
        for j in range(self.m + 1): # j = 0,1,2,...,m-1,m
            # 注意index是從上往下數，而price是從下往上算，所以用"m-j"
            f_matrix[self.m - j, self.n] = self.get_payoff(self.Smin + j * self.delta_S)

        # boundary condition
        f_matrix[0, :] = f_matrix[0, self.n]  # uppermost nodes = Smax
        f_matrix[self.m, :] = f_matrix[self.m, self.n]  # lowermost nodes = Smin

        # backward到前n-1期
        for i in range(self.n - 1, -1, -1):  # i = n-1,n-2,...,2,1,0

            # 已知的i + 1期整期的f，需要j = 0,1,2,...,m-1,m 共 m+1 期
            known_f_ls = f_matrix[0: m + 1, i + 1]  # j = 0,1,2,...,m-1,m
            ### 覆值時最保險的方法是在後面加上".copy()"，這樣在f_matrix上取值到known_f_ls時就不會更動原本f_matrix上的值 ###
            ### known_f_ls = f_matrix[0: m + 1, i + 1].copy()

            # matrix multiplication
            coeff_matrix = self.get_coefficient_matrix()
            unknown_f_ls = coeff_matrix @ known_f_ls

            # 將unknown_f_ls放到f_matrix的第i期
            if self.E_A == "A": # American
                for j in range(1, self.m):  # j = 1,2,...,m-1
                    # 注意index是從上往下數，而price是從下往上算，所以用"m-j"
                    exercise_value = max(self.get_payoff(self.Smin + (self.m - j) * self.delta_S), 0)
                    # 注意unknown_f_ls的index是從0開始
                    intrinsic_value = unknown_f_ls[j - 1] # j - 1 = 0,1,...,m-2
                    f_matrix[j, i] = max(exercise_value, intrinsic_value)
                    # To minimize the pricing error, f_matrix[j, i] must be positive.
                    f_matrix[j, i] = max(f_matrix[j, i], 0)

            elif self.E_A == "E": # European
                for j in range(1, self.m):  # j = 1,2,...,m-1
                    f_matrix[1: m, i] = unknown_f_ls
                    # To minimize the pricing error, f_matrix[j, i] must be positive.
                    f_matrix[1: m, i] = np.where(f_matrix[1: m, i] > 0, f_matrix[1: m, i], 0)


        # find index_j that meets index_j * delta_S = S0 by interpolation
        index_j = int(self.m * (self.S0 - self.Smin) / (self.Smax - self.Smin))  # 記得轉換為integer
        # 注意index是從上往下數，而option_value是從下往上算，所以用"m-j"
        target_j = self.m - index_j
        option_value = f_matrix[target_j, 0]

        return option_value



EuropeanCall = ExplicitFiniteDifference(S0, K, r, q, sigma, T, Smax, Smin, m, n, "C", "E")
European_call_option_value = EuropeanCall.backward_induction()
print(f"European call option value = {European_call_option_value:.4f}")

EuropeanPut = ExplicitFiniteDifference(S0, K, r, q, sigma, T, Smax, Smin, m, n, "P", "E")
European_put_option_value = EuropeanPut.backward_induction()
print(f"European put option value = {European_put_option_value:.4f}")

AmericanCall = ExplicitFiniteDifference(S0, K, r, q, sigma, T, Smax, Smin, m, n, "C", "A")
American_call_option_value = AmericanCall.backward_induction()
print(f"American call option value = {American_call_option_value:.4f}")

AmericanPut = ExplicitFiniteDifference(S0, K, r, q, sigma, T, Smax, Smin, m, n, "P", "A")
American_put_option_value = AmericanPut.backward_induction()
print(f"American put option value = {American_put_option_value:.4f}")










