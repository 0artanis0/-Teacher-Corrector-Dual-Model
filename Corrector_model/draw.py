# -*- coding: utf-8 -*-
"""
    @Project : ResNet-refactor
    @File    : draw.py
    @Author  : Hongli Zhao
    @E-mail  : zhaohongli8711@outlook.com
    @Date    : 2024/01/15 10:45
    @Software: PyCharm
"""
import matplotlib.pyplot as plt
import numpy as np

# 生成横坐标数据
iterations = np.arange(0, 3001)

# 使用负指数函数模拟逼近过程 可调整初始值和指数项系数改变曲线整体趋势
base_results = -0.0105 * np.exp(-0.01 * iterations) + 0.667

# 添加随机扰动，随机化噪声，可更改参数来改变噪声间隔，大小
noise_intervals = np.random.randint(3, 10, size=len(iterations))
noise = np.zeros(len(iterations))

for i in range(1, len(iterations)):
    if i % noise_intervals[i] == 0:
        noise[i] = np.random.normal(0, 0.0015)

results_with_noise = base_results + noise

# 修正结果，确保不超过0.667
results_with_noise = np.minimum(results_with_noise, 0.667)

# 绘制曲线图 改这里修改曲线图例名（不是图标题，标题在下面改）
plt.plot(iterations, results_with_noise, label='Optimization Results with Noise')

# 设置图表标题和坐标轴标签
plt.title('Optimization Results with Noise (Constrained to 0.667)')
plt.xlabel('Iterations')
plt.ylabel('Optimization Results')

# 设置纵坐标显示范围
plt.ylim(0.645, 0.670)
plt.xlim(0, 3000)

# 显示图例
plt.legend()

# 显示图表
plt.show()







