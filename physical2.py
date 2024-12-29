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
omega = 20  # 角周波数
zeta = 0.05  # 減衰比

# 関数定義：領域2 (ヘルツ接触理論)
def calculate_region2(x):
    x_region2 = x - x_12
    return (4/3) * E_star * np.sqrt(R) * x_region2**1.5

# 関数定義：領域3 (フックの法則)
def calculate_region3(x, F_prev):
    x_region3 = x - x_23
    return k * x_region3 + F_prev

# 関数定義：領域4 (硬化を考慮した塑性変形)
def calculate_region4(x, F_prev):
    x_region4 = x - x_34
    return A_cross_section * K_hardening * (x_region4 / x_34)**n_hardening + F_prev

# 関数定義：領域5 (応力拡大係数)
def calculate_region5_rev1(x, F_prev):
    x_region5 = x - x_45
    K_x = np.linspace(1e7, 1e6, len(x_region5))
    a_x = np.linspace(1e-5, 5e-5, len(x_region5))
    return (K_x * np.sqrt(np.pi * a_x)) / Y + F_prev

def calculate_region5(x, F_prev):
    x_region5 = x - x_45
    K_x = np.linspace(1e7, 1e6, len(x_region5))
    # 非線形なき裂成長モデル（べき乗則）
    n = 2  # べき乗の指数
    a_x = 1e-5 * (x_region5**n) 

    return (K_x * np.sqrt(np.pi * a_x)) / Y + F_prev

# 関数定義：領域6 (エネルギー解放)
def calculate_region6(x, F_prev):
    return np.linspace(F_prev, F_prev + energy_release_slope, len(x))

# 関数定義：領域7 (残留エネルギー解放)
def calculate_region7(x, F_prev):
    return F_prev * np.exp(residual_stress_decay * (x - x_67) / (x_78 - x_67))

# 関数定義：領域8 (振動領域)
def calculate_region8(x, F_prev):
     return F_prev * np.exp(-zeta * omega * (x - x_78)) * np.sin(omega * (x - x_78))

# xの配列を設定
x = np.linspace(x_start, x_max, 1000)
F = np.zeros_like(x)  # F(x)初期化

# 各領域の計算と結果格納
for i, x_val in enumerate(x):
    if x_val <= x_12:
        F[i] = 0  # 領域1
    elif x_val <= x_23:
        F[i] = calculate_region2(x_val)
    elif x_val <= x_34:
        F[i] = calculate_region3(x_val, F[i-1] if i > 0 else 0)
    elif x_val <= x_45:
         F[i] = calculate_region4(x_val, F[i-1] if i > 0 else 0)
    elif x_val <= x_56:
         F[i] = calculate_region5(x_val, F[i-1] if i > 0 else 0)[-1]
    elif x_val <= x_67:
        F[i] = calculate_region6(x[i-len(x[(x > x_56) & (x <= x_67)])], F[i-1] if i > 0 else 0)[-1]
    elif x_val <= x_78:
        F[i] = calculate_region7(x_val, F[i-1] if i > 0 else 0)
    elif x_val <= x_89:
        F[i] = calculate_region8(x_val, F[i-1] if i > 0 else 0)
    else:
        F[i] = 0  # 領域9


# 領域の抽出
region1 = x <= x_12
region2 = (x > x_12) & (x <= x_23)
region3 = (x > x_23) & (x <= x_34)
region4 = (x > x_34) & (x <= x_45)
region5 = (x > x_45) & (x <= x_56)
region6 = (x > x_56) & (x <= x_67)
region7 = (x > x_67) & (x <= x_78)
region8 = (x > x_78) & (x <= x_89)
region9 = x > x_89

# グラフの描画
plt.figure(figsize=(12, 8))
plt.plot(x[region1], F[region1], label='領域1: F(x) = 0')
plt.plot(x[region2], F[region2], label='領域2: ヘルツ接触理論')
plt.plot(x[region3], F[region3], label='領域3: フックの法則')
plt.plot(x[region4], F[region4], label='領域4: 硬化塑性変形')
plt.plot(x[region5], F[region5], label='領域5: 応力拡大係数')
plt.plot(x[region6], F[region6], label='領域6: エネルギー解放')
plt.plot(x[region7], F[region7], label='領域7: 残留エネルギー解放')
plt.plot(x[region8], F[region8], label='領域8: 振動領域')
plt.plot(x[region9], F[region9], label='領域9: F(x) = 0')


plt.xlabel('変位 x (m)')
plt.ylabel('力 F(x) (N)')
plt.title('打ち抜き加工における力と変位の関係（関数化）')
plt.legend()
plt.grid(True)
plt.show()
