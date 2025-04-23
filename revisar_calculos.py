import numpy as np
import matplotlib.pyplot as plt

# === Constants ===
c = 3e8

# === URBAN PARAMETERS (UMi - 3.5 GHz) ===
fc_urban = 3.5  # GHz
hBS_urban = 18
hUT_urban = 1.5
he_urban = 1
PLmax_DL_urban = 150
PLmax_UL_urban = 141

def calc_dbp(fc, hBS, hUT, he=1):
    return (4 * (hBS - he) * (hUT - he) * fc * 1e9) / c

dbp = calc_dbp(fc_urban, hBS_urban, hUT_urban, he_urban)

def PL_umi_los(d_m, fc, dbp, hBS, hUT):
    A = 32.4 + 20 * np.log10(fc) - 9.5 * np.log10(dbp**2 + (hBS - hUT)**2)
    return A + 40 * np.log10(d_m)

def PL_umi_nlos(d_m, fc, hUT):
    return 22.4 + 21.3 * np.log10(fc) - 0.3 * (hUT - 1.5) + 35.3 * np.log10(d_m)

def find_max_distance(pl_array, d_array, PLmax):
    idx = np.where(pl_array <= PLmax)[0]
    return d_array[idx[-1]] if len(idx) > 0 else None

# === More precise distance array ===
d_urban_m = np.linspace(10, 8500, 10000)
pl_los_urban = PL_umi_los(d_urban_m, fc_urban, dbp, hBS_urban, hUT_urban)
pl_nlos_urban = PL_umi_nlos(d_urban_m, fc_urban, hUT_urban)

# === Urban link distance limits ===
d_los_dl = find_max_distance(pl_los_urban, d_urban_m, PLmax_DL_urban)
d_nlos_dl = find_max_distance(pl_nlos_urban, d_urban_m, PLmax_DL_urban)
d_los_ul = find_max_distance(pl_los_urban, d_urban_m, PLmax_UL_urban)
d_nlos_ul = find_max_distance(pl_nlos_urban, d_urban_m, PLmax_UL_urban)

# === Console output ===

print("\n=== Distancias máximas urbanas (UMi - 3.5 GHz) ===")
print(f"{'Downlink':<10} - {'LoS':<4} : {d_los_dl:7.1f} m")
print(f"{'Downlink':<10} - {'NLoS':<4} : {d_nlos_dl:7.1f} m")
print(f"{'Uplink':<10} - {'LoS':<4} : {d_los_ul:7.1f} m")
print(f"{'Uplink':<10} - {'NLoS':<4} : {d_nlos_ul:7.1f} m")

# === URBAN PLOT ===
plt.figure(figsize=(10, 6))
plt.plot(d_urban_m, pl_los_urban, label="UMi LoS (3.5 GHz)", color='green')
plt.plot(d_urban_m, pl_nlos_urban, label="UMi NLoS (3.5 GHz)", color='orange')
plt.axhline(PLmax_DL_urban, linestyle='--', color='red', label="PL max DL")
plt.axhline(PLmax_UL_urban, linestyle='--', color='purple', label="PL max UL")
plt.axvline(d_los_dl, linestyle=':', color='green', label=f"DL LoS max = {d_los_dl:.0f} m")
plt.axvline(d_nlos_dl, linestyle=':', color='orange', label=f"DL NLoS max = {d_nlos_dl:.0f} m")
plt.axvline(d_los_ul, linestyle=':', color='green', alpha=0.5, label=f"UL LoS max = {d_los_ul:.0f} m")
plt.axvline(d_nlos_ul, linestyle=':', color='orange', alpha=0.5, label=f"UL NLoS max = {d_nlos_ul:.0f} m")
plt.title("Urban Propagation Model (UMi - 3.5 GHz)")
plt.xlabel("Distance (m)")
plt.ylabel("Path Loss (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === RURAL PARAMETERS (Okumura-Hata - 700 MHz) ===
fc_rural = 700  # MHz
hBS_rural = 20
hUT_rural = 1.5
PLmax_DL_rural = 150
PLmax_UL_rural = 141

def a_h(fc, hUT):
    logf = np.log10(fc)
    return (1.1 * logf - 0.7) * hUT - (1.56 * logf - 0.8)

def PL_rural(d_km, fc, hBS, hUT):
    logf = np.log10(fc)
    loghBS = np.log10(hBS)
    logd = np.log10(d_km)
    ah = a_h(fc, hUT)
    PL_urban = (
        69.55 + 26.16 * logf - 13.82 * loghBS + (44.9 - 6.55 * loghBS) * logd - ah
    )
    adjustment = -4.78 * (logf)**2 + 18.33 * logf - 40.94
    return PL_urban + adjustment

d_rural_km = np.linspace(0.1, 100, 10000)
d_rural_m = d_rural_km * 1000
pl_rural = PL_rural(d_rural_km, fc_rural, hBS_rural, hUT_rural)

d_max_dl_rural_km = find_max_distance(pl_rural, d_rural_km, PLmax_DL_rural)
d_max_ul_rural_km = find_max_distance(pl_rural, d_rural_km, PLmax_UL_rural)

# === Console output ===
print("\n=== Distancias máximas rurales (Okumura-Hata - 700 MHz) ===")
print(f"DL max: {d_max_dl_rural_km * 1000:.2f} m")
print(f"UL max: {d_max_ul_rural_km * 1000:.2f} m")

# === RURAL PLOT ===
plt.figure(figsize=(10, 6))
plt.plot(d_rural_m, pl_rural, label="Okumura-Hata Rural (700 MHz)", color='blue')
plt.axhline(PLmax_DL_rural, linestyle='--', color='red', label="PL max DL")
plt.axhline(PLmax_UL_rural, linestyle='--', color='purple', label="PL max UL")
plt.axvline(d_max_dl_rural_km * 1000, linestyle=':', color='red', label=f"DL max = {d_max_dl_rural_km * 1000:.0f} m")
plt.axvline(d_max_ul_rural_km * 1000, linestyle=':', color='purple', label=f"UL max = {d_max_ul_rural_km * 1000:.0f} m")
plt.title("Rural Propagation Model (Okumura-Hata - 700 MHz)")
plt.xlabel("Distance (m)")
plt.ylabel("Path Loss (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
