# 必要なライブラリをインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# CSVファイルのロード
file_path = "/mnt/data/シミュレーション波形2.csv"
data = pd.read_csv(file_path)

# 領域の境界パラメータの設定
x_start = -1.0  
x_12 = 0.0  
x_23 = 0.05  
x_34 = 0.6  
x_45 = 9.0  

# 領域1: F(x) = 0
region1_x = np.linspace(x_start, x_12, 100)
region1_y = np.zeros_like(region1_x)

# ヘルツ接触理論のモデル関数
def hertz_contact_model_fixed_R(x, E_star):
    R_fixed = 0.005  # 固定された曲率半径
    return (4 / 3) * E_star * np.sqrt(R_fixed) * (x - x_12)**1.5

# 領域2: フィッティング
region2_data = data[(data['Displacement (um)'] >= x_12) & (data['Displacement (um)'] <= x_23)]
popt, _ = curve_fit(hertz_contact_model_fixed_R, region2_data['Displacement (um)'], region2_data['Load force / punch circumference (N/mm)'])
E_star_fitted = popt[0]

# 領域2のデータ生成
region2_x = np.linspace(x_12, x_23, 100)
region2_y_fitted = hertz_contact_model_fixed_R(region2_x, E_star_fitted)

# フックの法則のモデル関数
def hooke_law_model(x, k):
    return k * (x - x_23) + region2_y_fitted[-1]

# 領域3: フィッティング
region3_data = data[(data['Displacement (um)'] >= x_23) & (data['Displacement (um)'] <= x_34)]
popt, _ = curve_fit(hooke_law_model, region3_data['Displacement (um)'], region3_data['Load force / punch circumference (N/mm)'])
k_fitted = popt[0]

# 領域3のデータ生成
region3_x = np.linspace(x_23, x_34, 100)
region3_y_fitted = hooke_law_model(region3_x, k_fitted)

# 硬化塑性変形のモデル関数
A_cross_section = 0.011  # 断面積
def plastic_hardening_model(x, K_hardening, n_hardening):
    return A_cross_section * K_hardening * ((x - x_34) / x_34)**n_hardening + region3_y_fitted[-1]

# 領域4: フィッティング
region4_data = data[(data['Displacement (um)'] >= x_34) & (data['Displacement (um)'] <= x_45)]
initial_guess = [2100, 0.23]
popt, _ = curve_fit(plastic_hardening_model, region4_data['Displacement (um)'], region4_data['Load force / punch circumference (N/mm)'], p0=initial_guess)
K_hardening_fitted, n_hardening_fitted = popt

# 領域4のデータ生成
region4_x = np.linspace(x_34, x_45, 100)
region4_y_fitted = plastic_hardening_model(region4_x, K_hardening_fitted, n_hardening_fitted)

# グラフの描画
plt.figure(figsize=(12, 6))

plt.plot(data['Displacement (um)'], data['Load force / punch circumference (N/mm)'], label='CSV Data', linewidth=2)
plt.plot(region1_x, region1_y, label='Region 1: F(x) = 0', linestyle='--', color='red')
plt.plot(region2_x, region2_y_fitted, label='Region 2: Hertz Contact (Fitted)', linestyle='--', color='green')
plt.plot(region3_x, region3_y_fitted, label='Region 3: Hooke\'s Law (Fitted)', linestyle='--', color='blue')
plt.plot(region4_x, region4_y_fitted, label='Region 4: Plastic Hardening (Fitted)', linestyle='--', color='purple')

plt.xlabel('Displacement (µm)')
plt.ylabel('Load Force / Punch Circumference (N/mm)')
plt.title('CSV Data with Regions 1 to 4 (Fitted) Overlay')
plt.legend()
plt.grid(True)
plt.show()

# フィットされたパラメータの出力
print(f"E* (Fitted): {E_star_fitted}")
print(f"k (Fitted): {k_fitted}")
print(f"K_hardening (Fitted): {K_hardening_fitted}")
print(f"n_hardening (Fitted): {n_hardening_fitted}")