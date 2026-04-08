import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 定义输入输出变量
# 温度偏差
error = ctrl.Antecedent(np.arange(-20, 21, 1), 'error')
# 加热功率
power = ctrl.Consequent(np.arange(0, 101, 1), 'power')

# 定义模糊集及其隶属函数
# 偏差的模糊集
error['NB'] = fuzz.trapmf(error.universe, [-20, -20, -15, -5])
error['NS'] = fuzz.trimf(error.universe, [-10, -5, 0])
error['Z'] = fuzz.trimf(error.universe, [-5, 0, 5])
error['PS'] = fuzz.trimf(error.universe, [0, 5, 10])
error['PB'] = fuzz.trapmf(error.universe, [5, 15, 20, 20])

# 功率的模糊集
power['OFF'] = fuzz.trapmf(power.universe, [0, 0, 10, 20])  # 确保 OFF 严格对应功率为 0
power['L'] = fuzz.trimf(power.universe, [10, 25, 40])
power['M'] = fuzz.trimf(power.universe, [30, 50, 70])
power['H'] = fuzz.trimf(power.universe, [60, 75, 90])
power['FULL'] = fuzz.trapmf(power.universe, [80, 90, 100, 100])

# 定义模糊规则
rule1 = ctrl.Rule(error['NB'], power['OFF'])
rule2 = ctrl.Rule(error['NS'], power['L'])
rule3 = ctrl.Rule(error['Z'], power['M'])
rule4 = ctrl.Rule(error['PS'], power['H'])
rule5 = ctrl.Rule(error['PB'], power['FULL'])

# 创建模糊控制系统
power_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
power_sim = ctrl.ControlSystemSimulation(power_ctrl)

# 不同温度的测试数据
test_temperatures = [60, 55, 50, 45, 40, 35, 30, 25, 20]

# 计算并输出不同温度下的控制结果
results = []
for T in test_temperatures:
    # 正确计算温度偏差
    e = 40 - T
    power_sim.input['error'] = e
    try:
        power_sim.compute()
        u = power_sim.output['power']
        results.append((T, u))
        print(f"当前温度: {T}°C, 温度偏差: {e}°C, 加热功率: {u:.2f}%")
    except Exception as ex:
        print(f"计算时出现错误: {ex}")

# 可视化验证
# 绘制偏差的隶属函数
fig, ax = plt.subplots()
error.view(ax=ax)
plt.title('偏差的隶属函数')
plt.show()

# 绘制功率的隶属函数
fig, ax = plt.subplots()
power.view(ax=ax)
plt.title('功率的隶属函数')
plt.show()

# 绘制不同温度下的控制结果
temperatures = [res[0] for res in results]
powers = [res[1] for res in results]
plt.plot(temperatures, powers, marker='o')
plt.xlabel('当前温度 (°C)')
plt.ylabel('加热功率 (%)')
plt.title('不同温度下的加热功率')
plt.grid(True)
plt.show()
