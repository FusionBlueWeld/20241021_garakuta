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
x_56 = 15.3  # 修正されたx_56

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

region2_x = np.linspace(x_12, x_23, 100)
region2_y_fitted = hertz_contact_model_fixed_R(region2_x, E_star_fitted)

# フックの法則のモデル関数
def hooke_law_model(x, k):
    return k * (x - x_23) + region2_y_fitted[-1]

# 領域3: フィッティング
region3_data = data[(data['Displacement (um)'] >= x_23) & (data['Displacement (um)'] <= x_34)]
popt, _ = curve_fit(hooke_law_model, region3_data['Displacement (um)'], region3_data['Load force / punch circumference (N/mm)'])
k_fitted = popt[0]

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

region4_x = np.linspace(x_34, x_45, 100)
region4_y_fitted = plastic_hardening_model(region4_x, K_hardening_fitted, n_hardening_fitted)

# 領域5のモデル関数
def region5_model(x, K_start, K_end, a_start, a_end, Y):
    K_x = np.linspace(K_start, K_end, len(x))
    a_x = np.maximum(np.linspace(a_start, a_end, len(x)), 1e-8)
    return (K_x * np.sqrt(np.pi * a_x)) / Y

# 領域5: 初期推定値
K_start_init, K_end_init, a_start_init, a_end_init = 1e7, 1e6, 1e-5, 5e-5
Y_init = 10000

region5_x = np.linspace(x_45, x_56, 100)
region5_data = data[(data['Displacement (um)'] >= x_45) & (data['Displacement (um)'] <= x_56)]

# ブロック処理: offset_45適用とフィッティングの繰り返し処理
def repeated_fitting_and_offset(iterations=5):
    Y_fitted_values = []
    region5_y_fitted_adjusted = None

    for i in range(iterations):
        # フィッティング処理
        popt, _ = curve_fit(
            lambda x, Y: region5_model(x, K_start_init, K_end_init, a_start_init, a_end_init, Y),
            region5_data['Displacement (um)'], 
            region5_data['Load force / punch circumference (N/mm)'],
            p0=[Y_init]
        )
        Y_fitted = popt[0]
        Y_fitted_values.append(Y_fitted)

        # 領域5のデータ生成
        region5_y_fitted = region5_model(region5_x, K_start_init, K_end_init, a_start_init, a_end_init, Y_fitted)

        # offset_45の適用と調整
        offset_45 = region5_y_fitted[0] - region4_y_fitted[-1]
        region5_y_fitted_adjusted = region5_y_fitted - offset_45

    return region5_y_fitted_adjusted, Y_fitted_values

# ブロック処理を5回繰り返し実行
region5_y_fitted_adjusted, Y_fitted_values = repeated_fitting_and_offset(iterations=5)

# グラフの描画
plt.figure(figsize=(12, 6))

plt.plot(data['Displacement (um)'], data['Load force / punch circumference (N/mm)'], label='CSV Data', linewidth=2)
plt.plot(region1_x, region1_y, label='Region 1: F(x) = 0', linestyle='--', color='red')
plt.plot(region2_x, region2_y_fitted, label='Region 2: Hertz Contact (Fitted)', linestyle='--', color='green')
plt.plot(region3_x, region3_y_fitted, label='Region 3: Hooke\'s Law (Fitted)', linestyle='--', color='blue')
plt.plot(region4_x, region4_y_fitted, label='Region 4: Plastic Hardening (Fitted)', linestyle='--', color='purple')
plt.plot(region5_x, region5_y_fitted_adjusted, label='Region 5: Fitted (Adjusted, Iterated)', linestyle='--', color='orange')

plt.xlabel('Displacement (µm)')
plt.ylabel('Load Force / Punch Circumference (N/mm)')
plt.title('CSV Data with Adjusted and Iterated Fitted Regions 1 to 5')
plt.legend()
plt.grid(True)
plt.show()

# 各フィッティング結果の出力
E_star_fitted, k_fitted, K_hardening_fitted, n_hardening_fitted, Y_fitted_values