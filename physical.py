# 必要なライブラリをインポート
import numpy as np
import matplotlib.pyplot as plt

# パラメータの設定
x_start = -1.0  
x_12 = 0.0  
x_23 = 0.05  
x_34 = 0.6  
x_45 = 9.0  
x_56 = 18.5  
x_67 = 18.71  
x_78 = 25.0  
x_89 = 30.0  
x_max = 40.0  

E_star = 900  # 有効弾性率
R = 0.01  # 曲率半径
k = 34  # ばね定数
A_cross_section = 0.011  # 断面積
K_hardening = 2100  # 硬化係数
n_hardening = 0.23  # 硬化指数
Y = 15000  # 形状係数
energy_release_slope = -25  # エネルギー解放の直線傾き
residual_stress_decay = -5  # 残留応力の減衰係数

# xの配列を設定
x = np.linspace(x_start, x_max, 1000)
F = np.zeros_like(x)  # F(x)初期化

# 領域1: F(x) = 0
region1 = x <= x_12
F[region1] = 0

# 領域2: ヘルツ接触理論
region2 = (x > x_12) & (x <= x_23)
x_region2 = x[region2] - x_12
F[region2] = (4/3) * E_star * np.sqrt(R) * x_region2**1.5

# 領域3: フックの法則
region3 = (x > x_23) & (x <= x_34)
x_region3 = x[region3] - x_23
F[region3] = k * x_region3 + F[region2][-1]

# 統合された領域4: 硬化を考慮した塑性変形
region4 = (x > x_34) & (x <= x_45)
x_region4 = x[region4] - x_34
F[region4] = A_cross_section * K_hardening * (x_region4 / x_34)**n_hardening + F[region3][-1]

# 領域5: 応力拡大係数
region5 = (x > x_45) & (x <= x_56)
x_region5 = x[region5] - x_45
K_x = np.linspace(1e7, 1e6, len(x_region5))
a_x = np.linspace(1e-5, 5e-5, len(x_region5))
F_model1 = (K_x * np.sqrt(np.pi * a_x)) / Y + F[region4][-1]

# 領域6: エネルギー解放
region6 = (x > x_56) & (x <= x_67)
F[region6] = np.linspace(F_model1[-1], F_model1[-1] + energy_release_slope, len(x[region6]))

# 領域7: 残留エネルギー解放
region7 = (x > x_67) & (x <= x_78)
F[region7] = F[region6][-1] * np.exp(residual_stress_decay * (x[region7] - x_67) / (x_78 - x_67))

# 領域8: 振動領域（減衰サインカーブ）
region8 = (x > x_78) & (x <= x_89)
omega = 20  # 角周波数
zeta = 0.05  # 減衰比
F[region8] = F[region7][-1] * np.exp(-zeta * omega * (x[region8] - x_78)) * np.sin(omega * (x[region8] - x_78))

# 領域9: F(x) = 0
region9 = x > x_89
F[region9] = 0

# グラフの描画
plt.figure(figsize=(12, 8))
plt.plot(x[region1], F[region1], label='領域1: F(x) = 0')
plt.plot(x[region2], F[region2], label='領域2: ヘルツ接触理論')
plt.plot(x[region3], F[region3], label='領域3: フックの法則')
plt.plot(x[region4], F[region4], label='領域4: 硬化塑性変形')
plt.plot(x[region5], F_model1, label='領域5: 応力拡大係数')
plt.plot(x[region6], F[region6], label='領域6: エネルギー解放')
plt.plot(x[region7], F[region7], label='領域7: 残留エネルギー解放')
plt.plot(x[region8], F[region8], label='領域8: 振動領域')
plt.plot(x[region9], F[region9], label='領域9: F(x) = 0')

plt.xlabel('変位 x (m)')
plt.ylabel('力 F(x) (N)')
plt.title('打ち抜き加工における力と変位の関係（エネルギー解放修正後）')
plt.legend()
plt.grid(True)
plt.show()