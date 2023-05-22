import numpy as np
from scipy.linalg import inv    ### 用scipy.linalg的話速度會太慢 ###
import time

start = time.time()

# input variable
S0 = 50
K = 50
r = 0.05
q = 0.01
sigma = 0.4
T = 0.5
Smax = 100
Smin = 0

m = 400 # for stock price partition
n = 100 # for time partition


class ImplicitFiniteDifference:

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
        a_ls = np.zeros(self.m + 1)
        b_ls = np.zeros(self.m + 1)
        c_ls = np.zeros(self.m + 1)
        for j in range(self.m + 1):
            a_ls[j] = (self.r - self.q)/2 * j * self.delta_T - self.sigma ** 2 * j ** 2 * self.delta_T / 2
            b_ls[j] = 1 + self.sigma ** 2 * j ** 2 * self.delta_T + self.r * self.delta_T
            c_ls[j] = - (self.r - self.q)/2 * j * self.delta_T - self.sigma ** 2 * j ** 2 * self.delta_T / 2


        coeff_matrix = np.zeros((self.m - 1, self.m - 1))
        # 先放係數矩陣中的第一列和最後一列
        # 第一列只有最前面兩個元素有值
        c_m_1 = c_ls[self.m - 1] # to adjust the first element of known_f_ls before solving the system of linear equations
        coeff_matrix[0, 0] = b_ls[self.m - 1]
        coeff_matrix[0, 1] = a_ls[self.m - 1]
        # 最後一列只有最後面兩個元素有值
        coeff_matrix[self.m - 2, self.m - 3] = c_ls[1]
        coeff_matrix[self.m - 2, self.m - 2] = b_ls[1]
        a1 = a_ls[1] # to adjust the last element of known_f_ls before solving the system of linear equations

        # 再放係數矩陣中的第二列到第m-2列
        for j in range(1, self.m - 2):  # j = 1,2,...,m-4, m-3
            for k in range(j - 1, j + 2):  # k = j-1, j, j+1
                # 注意計算a,b,c時的j是如講義中由下往上數的
                # coeff_matrix的index是從上往下數的，所以 m - j - 1
                if k == j - 1:
                    coeff_matrix[j, k] = c_ls[self.m - j - 1]
                elif k == j:
                    coeff_matrix[j, k] = b_ls[self.m - j - 1]
                elif k == j + 1:
                    coeff_matrix[j, k] = a_ls[self.m - j - 1]


        return coeff_matrix, c_m_1, a1


    def get_payoff(self, price):
        if self.C_P == "C": # call
            payoff = max(price - self.K, 0)
        elif self.C_P == "P": # put
            payoff = max(self.K - price, 0)

        return payoff

    def backward_induction(self):
        f_matrix = np.zeros((self.m + 1, self.n + 1))

        # boundary condition
        for i in range(0, self.n): # i = 0,1,2,...,n-2,n-1
            # uppermost nodes = Smax
            f_matrix[0, i] = self.get_payoff(self.Smax)
            # lowermost nodes = Smin
            f_matrix[self.m, i] = self.get_payoff(self.Smin)

        # 先算maturity payoff
        for j in range(self.m + 1): # j = 0,1,2,...,m-1,m
            # 注意index是從上往下數，而price是從下往上算，所以用"m-j"
            f_matrix[self.m - j, self.n] = self.get_payoff(self.Smin + j * self.delta_S)

        coeff_matrix = self.get_coefficient_matrix()[0]
        c_m_1 = self.get_coefficient_matrix()[1]
        a1 = self.get_coefficient_matrix()[2]

        # backward到前n-1期
        for i in range(self.n - 1, -1, -1): # i = n-1,n-2,...,2,1,0

            # 已知的i + 1期整期的f，只需要中間j = 1,2,...,m-2,m-1 共 m-1 期
            known_f_ls = f_matrix[1: m, i + 1] # j = 1,2,...,m-1
            ### 覆值時最保險的方法是在後面加上".copy()"，這樣在f_matrix上取值到known_f_ls時就不會更動原本f_matrix上的值 ###
            ### known_f_ls = f_matrix[1: m, i + 1].copy()

            # Remember to adjust the first and the last element.
            known_f_ls[0] = known_f_ls[0] - c_m_1 * f_matrix[0, i]
            known_f_ls[self.m -2] = known_f_ls[self.m - 2] - a1 * f_matrix[self.m, i]


            # solve the system of linear equations
            unknown_f_ls = inv(coeff_matrix) @ known_f_ls

            # 將unknown_f_ls放到f_matrix的第i期
            if self.E_A == "A": # American
                for j in range(1, self.m):  # j = 1,2,...,m-1
                    # 注意index是從上往下數，而price是從下往上算，所以用"m-j"
                    exercise_value = max(self.get_payoff(self.Smin + (self.m - j) * self.delta_S), 0)
                    # 注意unknown_f_ls的index是從0開始
                    intrinsic_value = unknown_f_ls[j - 1]
                    f_matrix[j, i] = max(exercise_value, intrinsic_value)
                    # To minimize the pricing error, f_matrix[j, i] must be positive.
                    f_matrix[j, i] = max(f_matrix[j, i], 0)

            elif self.E_A == "E": # European
                for j in range(1, self.m):  # j = 1,2,...,m-1
                    # 注意unknown_f_ls的index是從0開始
                    f_matrix[j, i] = unknown_f_ls[j - 1]
                    # To minimize the pricing error, f_matrix[j, i] must be positive.
                    f_matrix[j, i] = max(f_matrix[j, i], 0)


        # find index_j that meets index_j * delta_S = S0 by interpolation
        index_j = int(self.m * (self.S0 - self.Smin) / (self.Smax - self.Smin)) # 記得轉換為integer
        # 注意index是從上往下數，而option_value是從下往上算，所以用"m-j"
        target_j = self.m - index_j
        option_value = f_matrix[target_j, 0]

        return option_value





EuropeanCall = ImplicitFiniteDifference(S0, K, r, q, sigma, T, Smax, Smin, m, n, "C", "E")
European_call_option_value = EuropeanCall.backward_induction()
print(f"European call option value = {European_call_option_value:.4f}")

EuropeanPut = ImplicitFiniteDifference(S0, K, r, q, sigma, T, Smax, Smin, m, n, "P", "E")
European_put_option_value = EuropeanPut.backward_induction()
print(f"European put option value = {European_put_option_value:.4f}")

AmericanCall = ImplicitFiniteDifference(S0, K, r, q, sigma, T, Smax, Smin, m, n, "C", "A")
American_call_option_value = AmericanCall.backward_induction()
print(f"American call option value = {American_call_option_value:.4f}")

AmericanPut = ImplicitFiniteDifference(S0, K, r, q, sigma, T, Smax, Smin, m, n, "P", "A")
American_put_option_value = AmericanPut.backward_induction()
print(f"American put option value = {American_put_option_value:.4f}")

end = time.time()
print(end - start)







