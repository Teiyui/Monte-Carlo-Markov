import random
import numpy as np
import matplotlib.pyplot as plt


# 设置目标分布的密度函数为一个高斯分布
class Norm():
    def __init__(self, mean, std):
        self.mean = mean
        self.sigma = std

    # 一维高斯分布函数表达式
    def prob(self, x):
        return np.exp(-(x - self.mean) ** 2 / (2 * self.sigma ** 2.0)) * 1.0 / (np.sqrt(2 * np.pi) * self.sigma)

    # 返回numpy包封装好的采样分布进行对比
    def sample(self, num):
        sample = np.random.normal(self.mean, self.sigma, size=num)
        return sample


# 给定目标分布的密度函数：一个均值为0，方差为9的高斯分布
target_distribution = Norm(0, 3)


# 设置建议分布也为一个高斯分布
class Proposal():
    def __init__(self, std):
        self.sigma = std

    # 一维高斯分布函数表达式
    def prob(self, mean, x):
        return np.exp(-(x - mean) ** 2 / (2 * self.sigma ** 2.0)) * 1.0 / (np.sqrt(2 * np.pi) * self.sigma)

    # 返回一个采样结果
    def sampling(self, cur_mean):
        sample = np.random.normal(cur_mean, self.sigma, size=1)[0]
        return sample


# 给定的建议分布：一个均值为0，方差为25的高斯分布
proposal_distribution = Proposal(5)


# MH算法模型
# 收敛步数m，迭代步数n，目标函数p，建议分布q
def MHModel(m, n, p, q):
    # 任意选择一个初始值xi
    xi = random.uniform(0, 1.5)
    # 循环执行m+n次
    for i in range(m + n):
        # 从建议分布中随即采样一个候选状态xn
        xn = q.sampling(xi)
        # 计算接受概率
        accept = min(1, p.prob(xn) * q.prob(xn, xi) / (p.prob(xi) * q.prob(xi, xn)))
        # 从区间(0,1)中按均匀分布随机抽取一个数u
        u = random.uniform(0, 1)
        if u <= accept:
            xi = xn

        if i >= m:
            yield xi


m = 100
n = 100000
simulate_samples_p1 = [li for li in MHModel(m, n, target_distribution, proposal_distribution)]

plt.subplot(2, 2, 1)
plt.hist(simulate_samples_p1, 100)
plt.title("Simulated X ~ Norm(0/3)")

samples = target_distribution.sample(n)
plt.subplot(2, 2, 2)
plt.hist(samples, 100)
plt.title("True X ~ Norm(0/3)")

plt.show()


# 计算某个区间(a,b)的期望
def exceptional(simulate_samples, a, b):
    tmp1_samples = np.mat(simulate_samples)
    tmp2_samples = np.mat(simulate_samples)
    total_num = calculate_num(tmp1_samples, tmp2_samples, a, b)

    matrix = np.mat(simulate_samples)
    matrix[matrix <= a] = 0
    matrix[matrix >= b] = 0
    total_value = matrix.sum()

    return total_value / total_num


def calculate_num(tmp1, tmp2, a, b):
    tmp1[tmp1 > a] = 1
    tmp1[tmp1 <= a] = 0
    a_num = tmp1.sum()

    tmp2[tmp2 >= b] = 1
    tmp2[tmp2 < b] = 0
    b_num = tmp2.sum()

    return a_num - b_num


exceptional_value = exceptional(simulate_samples_p1, 1, 3)
print(exceptional_value)
