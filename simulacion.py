import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from shapely import GeometryCollection
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.prepared import prep
from unidecode import unidecode

# --- Ruta base ---
base_path = r"C:\Users\Diego Castro\Documents\Uvigo\UPM\Asignaturas\Primero\Proyecto REQM-SCOM\Simulacion"

# --- Cargar límites administrativos del País Vasco ---
alava = gpd.read_file(os.path.join(base_path, "BCN200_ALAVA", "ALAVA", "BCN200_0101S_LIM_ADM.shp"))
bizkaia = gpd.read_file(os.path.join(base_path, "BCN200_VIZCAYA", "VIZCAYA", "BCN200_0101S_LIM_ADM.shp"))
gipuzkoa = gpd.read_file(os.path.join(base_path, "BCN200_GUIPUZCOA", "GUIPUZCOA", "BCN200_0101S_LIM_ADM.shp"))
pais_vasco_lim = gpd.GeoDataFrame(pd.concat([alava, bizkaia, gipuzkoa], ignore_index=True))

# --- Cargar núcleos de población ---
pob_alava = gpd.read_file(os.path.join(base_path, "BCN200_ALAVA", "ALAVA", "BCN200_0501S_NUC_POB.shp"))
pob_vizcaya = gpd.read_file(os.path.join(base_path, "BCN200_VIZCAYA", "VIZCAYA", "BCN200_0501S_NUC_POB.shp"))
pob_gipuzkoa = gpd.read_file(os.path.join(base_path, "BCN200_GUIPUZCOA", "GUIPUZCOA", "BCN200_0501S_NUC_POB.shp"))
poblacion = pd.concat([pob_alava, pob_vizcaya, pob_gipuzkoa], ignore_index=True)
poblacion = gpd.GeoDataFrame(poblacion, geometry='geometry', crs=pob_alava.crs)

# --- Cargar radios personalizados desde CSV ---
csv_radios_path = r"C:\Users\Diego Castro\Documents\Uvigo\UPM\Asignaturas\Primero\Proyecto REQM-SCOM\Simulacion\Radios_por_Ciudad.csv"
df_radios = pd.read_csv(csv_radios_path)
radios_personalizados = dict(zip(df_radios['Ciudad'], df_radios['Radio_m']))

# --- Reproyección a CRS métrico ---
pais_vasco_lim = pais_vasco_lim.to_crs(epsg=25830)
poblacion = poblacion.to_crs(epsg=25830)

# --- Detectar población y clasificar ---
col_poblacion = [col for col in poblacion.columns if 'POB' in col.upper() or 'HAB' in col.upper()]
col_pob = col_poblacion[0]
poblacion[col_pob] = pd.to_numeric(poblacion[col_pob], errors='coerce')
poblacion = poblacion[poblacion[col_pob] > 500].copy()


def match_ciudad(nombre, radios_dict):
    nombre_limpio = unidecode(nombre.lower().strip())
    for ciudad in radios_dict:
        ciudad_limpia = unidecode(ciudad.lower().strip())
        if ciudad_limpia in nombre_limpio:
            return radios_dict[ciudad]
    return None


def clasificar_nucleo(pob):
    if pob >= 20000:
        return "urbano"
    elif pob >= 1000:
        return "semiurbano"
    else:
        return "rural"

poblacion['tipo_entorno'] = poblacion[col_pob].apply(clasificar_nucleo)

# --- Crear hexágonos ---
def crear_hexagono(centro, r):
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    coords = [(centro.x + r * np.cos(a), centro.y + r * np.sin(a)) for a in angles]
    return Polygon(coords)

def crear_hex_grid_en_poligono(poligono, radio, densidad=0.75, umbral_interseccion=0.1):
    bounds = poligono.bounds
    minx, miny, maxx, maxy = bounds
    dx = np.sqrt(3) * radio * densidad
    dy = 1.5 * radio * densidad
    hexagonos = []
    y = miny
    row = 0
    while y < maxy + dy:
        x_offset = 0 if row % 2 == 0 else dx / 2
        x = minx + x_offset
        while x < maxx + dx:
            centro = Point(x, y)
            hex = crear_hexagono(centro, radio)
            if poligono.intersects(hex):
                inter = poligono.intersection(hex)
                if inter.area / hex.area > umbral_interseccion:
                    hexagonos.append(hex)
            x += dx
        y += dy
        row += 1
    return hexagonos

# --- Radios de cobertura ---
tipos_radio = {
    'micro_urbana': 105,
    'macro_urbana': 1000,    # 🔧 reducido de 1290 → 800 para acercarnos a 500 celdas
    'macro_rural': 6141
}

# --- Funciones para grids y clasificaciones ---
def crear_hex_grid(radio, espaciado):
    minx, miny, maxx, maxy = pais_vasco_lim.total_bounds
    dx = np.sqrt(3) * radio * espaciado
    dy = 1.5 * radio * espaciado
    hex_centers = []
    y = miny
    row = 0
    while y < maxy + dy:
        x_offset = 0 if row % 2 == 0 else dx / 2
        x = minx + x_offset
        while x < maxx + dx:
            hex_centers.append(Point(x, y))
            x += dx
        y += dy
        row += 1
    return gpd.GeoDataFrame(geometry=hex_centers, crs=pais_vasco_lim.crs)

def clasificar_hex(pt, tipo):
    intersecta = poblacion[poblacion.geometry.intersects(pt.buffer(2500))]
    if intersecta.empty:
        # Considerar como rural si no hay población en el área
        return True if tipo == 'rural' else False
    tipo_pred = intersecta['tipo_entorno'].value_counts().idxmax()
    return tipo_pred == tipo

def generar_micro_urbanas_personalizado():
    micro_hexes = []
    for _, row in poblacion.iterrows():
        if row['tipo_entorno'] == 'urbano':
            nombre = str(row.get('ETIQUETA', '')).lower().strip()
            radio = None
            for ciudad in radios_personalizados:
                if ciudad.lower() in nombre:
                    radio = radios_personalizados[ciudad]
                    break
            if radio is None:
                radio = tipos_radio['micro_urbana']
            hexagonos = crear_hex_grid_en_poligono(
                row.geometry,
                radio,
                densidad=1,
                umbral_interseccion=0.1
            )
            micro_hexes.extend([{'tipo': 'micro_urbana', 'geometry': h} for h in hexagonos])
    return gpd.GeoDataFrame(micro_hexes, crs=poblacion.crs)

# --- Generar celdas ---
macro_rural = crear_hex_grid(tipos_radio['macro_rural'], 0.87)
macro_rural = macro_rural[macro_rural.geometry.within(pais_vasco_lim.unary_union)]
macro_rural = macro_rural[macro_rural.geometry.apply(lambda g: clasificar_hex(g, 'rural'))]
macro_rural['tipo'] = 'macro_rural'
macro_rural['geometry'] = macro_rural.geometry.apply(lambda g: crear_hexagono(g, tipos_radio['macro_rural']))

macro_urbana = crear_hex_grid(tipos_radio['macro_urbana'], 0.87)
macro_urbana = macro_urbana[macro_urbana.geometry.within(pais_vasco_lim.unary_union)]
macro_urbana = macro_urbana[macro_urbana.geometry.apply(lambda g: clasificar_hex(g, 'semiurbano'))]
macro_urbana['tipo'] = 'macro_urbana'
macro_urbana['geometry'] = macro_urbana.geometry.apply(lambda g: crear_hexagono(g, tipos_radio['macro_urbana']))

micro_urbana = generar_micro_urbanas_personalizado()

# --- Eliminar microceldas solapadas ---
macro_union = unary_union(pd.concat([macro_rural, macro_urbana])['geometry'].tolist())
micro_urbana = micro_urbana[~micro_urbana.geometry.apply(lambda g: prep(macro_union).contains(g))]

# --- Combinar ---
gdf_cobertura = pd.concat([macro_rural, macro_urbana, micro_urbana], ignore_index=True)

# --- Detectar y rellenar huecos ---
cobertura_total = unary_union(gdf_cobertura.geometry)
huecos = pais_vasco_lim.unary_union.difference(cobertura_total)
if isinstance(huecos, (Polygon, MultiPolygon)):
    huecos = [huecos] if isinstance(huecos, Polygon) else list(huecos.geoms)
elif isinstance(huecos, GeometryCollection):
    huecos = [g for g in huecos.geoms if isinstance(g, (Polygon, MultiPolygon))]
else:
    huecos = []

nuevos_hex_centers = []
dx = np.sqrt(3) * tipos_radio['macro_rural'] * 0.95
dy = 1.5 * tipos_radio['macro_rural'] * 0.95
minx, miny, maxx, maxy = pais_vasco_lim.total_bounds
y = miny
row = 0
while y < maxy + dy:
    x_offset = 0 if row % 2 == 0 else dx / 2
    x = minx + x_offset
    while x < maxx + dx:
        p = Point(x, y)
        if any(h.intersects(p) for h in huecos):
            nuevos_hex_centers.append(p)
        x += dx
    y += dy
    row += 1

hex_huecos = gpd.GeoDataFrame({
    'tipo': 'macro_rural',
    'geometry': [crear_hexagono(p, tipos_radio['macro_rural']) for p in nuevos_hex_centers]
}, crs=pais_vasco_lim.crs)


# Añadir a la cobertura total
gdf_cobertura = pd.concat([gdf_cobertura, hex_huecos], ignore_index=True)

conteo = gdf_cobertura['tipo'].value_counts()
print("\nNúmero de celdas por tipo:")
print(conteo)

# --- Visualización ---
fig, ax = plt.subplots(figsize=(12, 12))

print("Columnas del shapefile de núcleos de población:")
print(poblacion.columns)

#gdf_huecos_geom = gpd.GeoDataFrame(geometry=huecos, crs=pais_vasco_lim.crs)
#gdf_huecos_geom.plot(ax=ax, color='none', edgecolor='magenta', linewidth=1, linestyle='--', zorder=10)

pais_vasco_lim.boundary.plot(ax=ax, color='black', linewidth=0.5, zorder=1)
gdf_cobertura[gdf_cobertura['tipo'] == 'macro_rural'].plot(ax=ax, facecolor='lightblue', edgecolor='blue', alpha=0.35, linewidth=0.4, label="Macro Rural", zorder=2)
gdf_cobertura[gdf_cobertura['tipo'] == 'macro_urbana'].plot(ax=ax, color='orange', alpha=0.4, label="Macro Urbana", zorder=3)
gdf_cobertura[gdf_cobertura['tipo'] == 'micro_urbana'].plot(ax=ax, color='red', alpha=0.4, label="Micro Urbana", zorder=4)
poblacion.boundary.plot(ax=ax, color='black', linewidth=0.5, zorder=5)
plt.title("Cobertura 5G realista con celdas ajustadas y densificadas por entorno")
plt.legend(loc='lower left')
plt.axis('off')
plt.show()