import numpy as np
import matplotlib.pyplot as plt

# 材料モデル（Johnson-Cook）
class JohnsonCookMaterial:
    def __init__(self, A, B, n, C, m, rho, Cp, thermal_conductivity):
        # A: 初期降伏応力, B: 加工硬化係数, n: 加工硬化指数
        # C: ひずみ速度感受性係数, m: 温度感受性指数
        # rho: 密度, Cp: 比熱, thermal_conductivity: 熱伝導率
        self.A = A
        self.B = B
        self.n = n
        self.C = C
        self.m = m
        self.rho = rho
        self.Cp = Cp
        self.thermal_conductivity = thermal_conductivity

    def get_flow_stress(self, strain, strain_rate_eff, T, Tm):
        # 相当ひずみ、有効ひずみ速度、温度、融点から流動応力を計算
        epsilon_dot_ref = 1.0  # 基準ひずみ速度
        T_ref = 298.0  # 基準温度

        term1 = self.A + self.B * (strain)**self.n
        term2 = (1 + self.C * np.log(strain_rate_eff / epsilon_dot_ref))
        term3 = (1 - ((T - T_ref) / (Tm - T_ref))**self.m)
        return term1 * term2 * term3

# 摩擦モデル（Swift）
class SwiftFriction:
    def __init__(self, mu_0, a, b):
        # mu_0: 基礎摩擦係数, a, b: 係数
        self.mu_0 = mu_0
        self.a = a
        self.b = b

    def get_friction_coefficient(self, pressure, slip_velocity):
        # 面圧とすべり速度から摩擦係数を計算
        mu = self.mu_0 * (1 + self.a * np.exp(-self.b * pressure)) * (1 - np.exp(-self.c * slip_velocity)) # 修正が必要
        return mu_0 # 一旦、定数として返す

# 損傷モデル（Lemaitre）
class LemaitreDamage:
    def __init__(self, S0, b, Dcrit):
        # S0, b: 材料定数, Dcrit: 臨界損傷値
        self.S0 = S0
        self.b = b
        self.Dcrit = Dcrit
        self.D = 0.0 # 損傷変数

    def update_damage(self, equivalent_stress, delta_equivalent_strain):
        # 相当応力と相当ひずみ増分から損傷変数を更新
        if self.D < self.Dcrit:
            dD = (self.S0 * (equivalent_stress)**self.b / (1 - self.D)) * delta_equivalent_strain
            self.D += dD
        return self.D

    def get_damage(self):
        return self.D

# プレス加工シミュレーション
def simulate_punching(
    material, friction_model, damage_model,
    punch_diameter, material_thickness, clearance, ram_speed,
    punch_stiffness, die_stiffness,
    total_time, sampling_rate, Tm
):
    times = np.linspace(0, total_time, int(total_time * sampling_rate), endpoint=False)
    dt = times[1] - times[0]
    forces = np.zeros_like(times)
    penetration_depths = np.zeros_like(times)
    strains = np.zeros_like(times)
    strain_rates = np.zeros_like(times)
    temperatures = np.ones_like(times) * 298.0 # 初期温度

    damage = 0.0

    for i, t in enumerate(times):
        penetration_depth = ram_speed * t
        penetration_depths[i] = penetration_depth

        if i > 0:
            delta_penetration = penetration_depths[i] - penetration_depths[i-1]
        else:
            delta_penetration = 0.0

        # パンチとダイの変形を考慮
        punch_force = 0.0
        die_force = 0.0

        # 材料のひずみとひずみ速度を推定 (簡略化)
        strain_eff = penetration_depth / material_thickness
        strains[i] = strain_eff
        if i > 0:
            strain_rate_eff = (strains[i] - strains[i-1]) / dt
        else:
            strain_rate_eff = 0.0
        strain_rates[i] = strain_rate_eff

        # 流動応力を計算
        flow_stress = material.get_flow_stress(strains[i], strain_rates[i], temperatures[i], Tm)

        # 接触面積を推定
        contact_area = np.pi * (punch_diameter/2)**2

        # 摩擦力を計算 (Swiftモデルを使用)
        pressure = flow_stress # 近似
        slip_velocity = ram_speed # 近似
        friction_coefficient = friction_model.get_friction_coefficient(pressure, slip_velocity)
        friction_force = friction_coefficient * pressure * np.pi * punch_diameter * penetration_depth # 簡略化

        # 抵抗力を計算
        resisting_force = flow_stress * contact_area + friction_force

        # 金型の変形を考慮した荷重
        forces[i] = resisting_force # - punch_force - die_force # 金型の変形は今回は省略

        # 損傷を評価
        delta_strain = 0
        if i > 0:
            delta_strain = strains[i] - strains[i-1]

        damage = damage_model.update_damage(flow_stress, delta_strain)

        if damage >= damage_model.Dcrit:
            forces[i:] = forces[i-1] # 破断後の荷重を維持
            break

        # 熱発生と温度変化 (非常に簡略化)
        plastic_work = flow_stress * delta_strain # 塑性仕事
        friction_work = friction_force * delta_penetration # 摩擦仕事
        heat_generated = plastic_work + friction_work
        delta_T = heat_generated / (material.rho * material.Cp)
        temperatures[i] = temperatures[i-1] + delta_T if i > 0 else temperatures[i]

    return times, forces

# パラメータ設定
# 材料 (SPCC相当)
material_params = {
    'A': 180e6, 'B': 510e6, 'n': 0.26, 'C': 0.014, 'm': 1.1,
    'rho': 7850, 'Cp': 460, 'thermal_conductivity': 50
}
material = JohnsonCookMaterial(**material_params)

# 摩擦 (鋼-鋼)
friction_params = {'mu_0': 0.15, 'a': 0.5, 'b': 0.1}
friction_model = SwiftFriction(**friction_params)

# 損傷
damage_params = {'S0': 1000e6, 'b': 1.5, 'Dcrit': 0.8}
damage_model = LemaitreDamage(**damage_params)

# 加工条件
punch_diameter = 10e-3
material_thickness = 1.0e-3
clearance = 0.05 * punch_diameter
ram_speed = 1.0
punch_stiffness = 1e9 # N/m
die_stiffness = 1e9 # N/m
Tm = 1800 # 融点 (K)

# シミュレーション設定
total_time = 0.1
sampling_rate = 10000

# シミュレーション実行
times, forces = simulate_punching(
    material, friction_model, damage_model,
    punch_diameter, material_thickness, clearance, ram_speed,
    punch_stiffness, die_stiffness,
    total_time, sampling_rate, Tm
)

# プロット
plt.figure(figsize=(10, 6))
plt.plot(times, forces)
plt.xlabel("時間 [s]")
plt.ylabel("荷重 [N]")
plt.title("打ち抜き加工 荷重波形シミュレーション (高度モデル)")
plt.grid(True)
plt.show()