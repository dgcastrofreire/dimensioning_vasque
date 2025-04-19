import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# --- Cargar tus datos como ya hacías ---
# Ajusta tu ruta aquí:
base_path = r"C:\Users\Diego Castro\Documents\Uvigo\UPM\Asignaturas\Primero\Proyecto REQM-SCOM\Simulacion"

alava = gpd.read_file(os.path.join(base_path, "BCN200_ALAVA", "ALAVA", "BCN200_0101S_LIM_ADM.shp"))
bizkaia = gpd.read_file(os.path.join(base_path, "BCN200_VIZCAYA", "VIZCAYA", "BCN200_0101S_LIM_ADM.shp"))
gipuzkoa = gpd.read_file(os.path.join(base_path, "BCN200_GUIPUZCOA", "GUIPUZCOA", "BCN200_0101S_LIM_ADM.shp"))

pais_vasco_lim = gpd.GeoDataFrame(pd.concat([alava, bizkaia, gipuzkoa], ignore_index=True))

# --- Curvas de nivel ---
curvas_alava = gpd.read_file(os.path.join(base_path, "BCN200_ALAVA", "ALAVA", "BCN200_0202L_CURV_NIV.shp"))
curvas_bizkaia = gpd.read_file(os.path.join(base_path, "BCN200_VIZCAYA", "VIZCAYA", "BCN200_0202L_CURV_NIV.shp"))
curvas_gipuzkoa = gpd.read_file(os.path.join(base_path, "BCN200_GUIPUZCOA", "GUIPUZCOA", "BCN200_0202L_CURV_NIV.shp"))

curvas_vasco = pd.concat([curvas_alava, curvas_bizkaia, curvas_gipuzkoa], ignore_index=True)
col_altitud = [col for col in curvas_vasco.columns if 'COTA' in col.upper() or 'ELEV' in col.upper()]
altura_col = col_altitud[0]
curvas_vasco[altura_col] = pd.to_numeric(curvas_vasco[altura_col], errors='coerce')
curvas_vasco = gpd.GeoDataFrame(curvas_vasco, geometry='geometry', crs=curvas_alava.crs)

# --- Población ---
pob_alava = gpd.read_file(os.path.join(base_path, "BCN200_ALAVA", "ALAVA", "BCN200_0501S_NUC_POB.shp"))
pob_vizcaya = gpd.read_file(os.path.join(base_path, "BCN200_VIZCAYA", "VIZCAYA", "BCN200_0501S_NUC_POB.shp"))
pob_gipuzkoa = gpd.read_file(os.path.join(base_path, "BCN200_GUIPUZCOA", "GUIPUZCOA", "BCN200_0501S_NUC_POB.shp"))

poblacion = pd.concat([pob_alava, pob_vizcaya, pob_gipuzkoa], ignore_index=True)
poblacion = gpd.GeoDataFrame(poblacion, geometry='geometry', crs=pob_alava.crs)

# Columna de población
col_poblacion = [col for col in poblacion.columns if 'POB' in col.upper() or 'HAB' in col.upper()]
col_pob = col_poblacion[0]
poblacion[col_pob] = pd.to_numeric(poblacion[col_pob], errors='coerce')
poblacion = poblacion[poblacion[col_pob] > 500].copy()

# Escalar tamaño visual
poblacion['tamano'] = np.sqrt(poblacion[col_pob] + 50)

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

# Mapa 1: Orografía
curvas_vasco.plot(
    ax=ax1,
    column=altura_col,
    cmap='terrain',
    linewidth=0.5,
    legend=True,
    legend_kwds={'label': "Altitud (m)", 'shrink': 0.6}
)
pais_vasco_lim.boundary.plot(ax=ax1, color='grey', linewidth=0.3)
ax1.set_title("Mapa orográfico del País Vasco")
ax1.axis('off')

# Mapa 2: Población
pais_vasco_lim.boundary.plot(ax=ax2, color='lightgrey', linewidth=0.3)
poblacion.plot(
    ax=ax2,
    column=col_pob,
    cmap='YlOrRd',
    markersize=poblacion['tamano'],
    legend=True,
    legend_kwds={'label': "Habitantes por núcleo", 'shrink': 0.6}
)
ax2.set_title("Núcleos de población en el País Vasco")
ax2.axis('off')

plt.tight_layout()
plt.show()
