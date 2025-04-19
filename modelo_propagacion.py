import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# --- Parámetros generales ---
P_tx = 63         # dBm (EIRP en 3.5 GHz, según documento)
G_tx = 0          # Ganancia ya incluida en el EIRP
G_rx = 11         # Ganancia receptor
P_rx_min = -95    # Sensibilidad del receptor
PL_max = P_tx + G_tx + G_rx - P_rx_min

# --- Alturas ---
h_tx_rural = 20
h_tx_urbano = 18
h_tx_macro = 25
h_rx = 1.5
h_e = 1.0  # altura efectiva del entorno urbano

# --- Frecuencias ---
freq_rural = 700      # MHz
freq_macro_urb = 3500 # MHz
freq_micro = 3500     # MHz

# --- Constante ---
c = 3e8  # velocidad de la luz en m/s

# --- Corrección por altura del receptor para Hata ---
def A_h(h_rx, fc):
    return (1.1 * np.log10(fc) - 0.7) * h_rx - (1.56 * np.log10(fc) - 0.8)

# --- Modelo Okumura-Hata Rural ---
def PL_hata_rural(d_km, fc=freq_rural, h_tx=h_tx_rural, h_rx=h_rx):
    A = A_h(h_rx, fc)
    pl_urban = 69.55 + 26.16 * np.log10(fc) - 13.82 * np.log10(h_tx)
    pl_urban += (44.9 - 6.55 * np.log10(h_tx)) * np.log10(d_km) - A
    pl_rural = pl_urban - 4.78 * (np.log10(fc))**2 + 18.33 * np.log10(fc) - 40.94
    return pl_rural

# --- Modelo COST-231 Hata Urbana (Macrocelda) ---
def PL_cost231_macro(d_km, fc=freq_macro_urb, h_tx=h_tx_macro, h_rx=h_rx):
    A = A_h(h_rx, fc)
    C = 3  # Área metropolitana
    pl = 46.3 + 33.9 * np.log10(fc) - 13.82 * np.log10(h_tx)
    pl += (44.9 - 6.55 * np.log10(h_tx)) * np.log10(d_km) - A + C
    return pl

# --- Distancia de ruptura (3GPP TR 38.901) ---
def dbp_prime(fc=freq_micro, hBS=h_tx_urbano, hUT=h_rx):
    return (4 * (hBS - h_e) * (hUT - h_e) * fc * 1e9) / c

# --- Modelo UMi - Line-of-Sight (LoS) ---
def PL_umi_LOS(d_km, fc=freq_micro, hBS=h_tx_urbano, hUT=h_rx):
    d_m = d_km * 1000
    dbp = dbp_prime(fc, hBS, hUT)
    term = 32.4 + 40 * np.log10(d_m) + 20 * np.log10(fc)
    correction = 9.5 * np.log10(dbp**2 + (hBS - hUT)**2)
    return term - correction

# --- Modelo UMi - Non-Line-of-Sight (NLoS) ---
def PL_umi_NLOS(d_km, fc=freq_micro, hBS=h_tx_urbano, hUT=h_rx):
    d_m = d_km * 1000
    los = PL_umi_LOS(d_km, fc, hBS, hUT)
    pl_nlos_prime = 35.3 * np.log10(d_m) + 22.4 + 21.3 * np.log10(fc) - 0.3 * (hUT - 1.5)
    return np.maximum(los, pl_nlos_prime)

# --- Encontrar distancia máxima para un modelo dado ---
def encontrar_distancia_max(pl_model, label):
    d_min, d_max = 0.01, 30  # km
    pl_min = pl_model(d_min)
    pl_max_val = pl_model(d_max)

    if (pl_min - PL_max) * (pl_max_val - PL_max) > 0:
        print(f"⚠️ {label} - No hay cruce con PL_max en el rango [{d_min}, {d_max}] km")
        return None

    sol = root_scalar(lambda d: pl_model(d) - PL_max, bracket=[d_min, d_max], method='brentq')
    if sol.converged:
        print(f"✅ {label} - Distancia máxima: {sol.root*1000:.0f} m")
        return sol.root
    else:
        print(f"❌ {label} - No convergió")
        return None

# --- Calcular distancias máximas ---
d_rural = encontrar_distancia_max(PL_hata_rural, "Macrocelda Rural (Hata)")
d_macro = encontrar_distancia_max(PL_cost231_macro, "Macrocelda Urbana (COST-231)")
d_micro_LOS = encontrar_distancia_max(PL_umi_LOS, "Microcelda Urbana (UMi LOS)")
d_micro_NLOS = encontrar_distancia_max(PL_umi_NLOS, "Microcelda Urbana (UMi NLOS)")

# --- Visualización ---
distancias = np.linspace(0.05, 10, 400)
plt.figure(figsize=(10,6))
plt.plot(distancias*1000, PL_hata_rural(distancias), label="Macrocelda Rural - Hata")
plt.plot(distancias*1000, PL_cost231_macro(distancias), label="Macrocelda Urbana - COST-231")
plt.plot(distancias*1000, PL_umi_LOS(distancias), label="Microcelda - UMi LOS")
plt.plot(distancias*1000, PL_umi_NLOS(distancias), label="Microcelda - UMi NLOS")
plt.axhline(PL_max, color='k', linestyle='--', label="Límite PL")
plt.xlabel("Distancia (m)")
plt.ylabel("Path Loss (dB)")
plt.title("Modelos de Propagación por Tipo de Celda")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
