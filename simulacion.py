import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
from unidecode import unidecode
from collections import Counter
# --- Detectar y rellenar huecos en la cobertura ---
from shapely import GeometryCollection, MultiPolygon
import matplotlib.patches as mpatches

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
csv_radios_path = os.path.join(base_path, "radios_personalizados2.csv")
df_radios = pd.read_csv(csv_radios_path)
df_radios['Ciudad_limpia'] = df_radios['Ciudad'].apply(lambda x: unidecode(str(x).lower().strip()))
radios_personalizados = dict(zip(df_radios['Ciudad_limpia'], df_radios['Radio_m']))

# --- Reproyección ---
pais_vasco_lim = pais_vasco_lim.to_crs(epsg=25830)
poblacion = poblacion.to_crs(epsg=25830)

# --- Clasificación y limpieza ---
col_pob = [col for col in poblacion.columns if 'POB' in col.upper() or 'HAB' in col.upper()][0]
poblacion[col_pob] = pd.to_numeric(poblacion[col_pob], errors='coerce')
poblacion = poblacion[poblacion[col_pob] > 500].copy()

def clasificar_nucleo(pob):
    if pob >= 20000:
        return "urbano"
    else:
        return "rural"

poblacion['tipo_entorno'] = poblacion[col_pob].apply(clasificar_nucleo)
poblacion['ciudad_limpia'] = poblacion['ETIQUETA'].apply(lambda x: unidecode(str(x).lower().strip()))

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

densidades_personalizadas = {
    'bilbao': 0.73,
    'donostia/san sebastian': 0.57,
    'vitoria-gasteiz': 0.35,
    'irun': 0.54,
    'durango': 0.74,
    'laudio/llodio': 0.95,
    'errenteria': 0.48,
    'san vicente de barakaldo/san bizenti-barakaldo': 0.63,
    'arizgoiti': 1.5,
    'arrasate edo mondragon': 0.5,
    'eibar': 0.55,
    'portugalete': 1.05,
    'sestao': 1.2,
    'algorta': 0.7,
    'las arenas-areeta': 0.7,
    'zarautz': 0.65
    # añade más si quieres afinar aún más
}

def generar_micro_urbanas_personalizado():
    micro_hexes = []
    for _, row in poblacion.iterrows():
        if row['tipo_entorno'] == 'urbano':
            ciudad_limpia = unidecode(str(row['ETIQUETA']).lower().strip())
            radio = radios_personalizados.get(ciudad_limpia, 500)
            densidad = densidades_personalizadas.get(ciudad_limpia, 0.75)  # por defecto

            hexagonos = crear_hex_grid_en_poligono(
                row.geometry,
                radio,
                densidad=densidad,
                umbral_interseccion=0.001
            )

            micro_hexes.extend([{'tipo': 'micro_urbana', 'ciudad': ciudad_limpia, 'geometry': h} for h in hexagonos])
    return gpd.GeoDataFrame(micro_hexes, crs=poblacion.crs)

# --- Generar macro_rural ---
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

def clasificar_hex(pt):
    intersecta = poblacion[poblacion.geometry.intersects(pt.buffer(2500))]
    if intersecta.empty:
        return True
    tipo_pred = intersecta['tipo_entorno'].value_counts().idxmax()
    return tipo_pred == 'rural'

macro_rural = crear_hex_grid(14770, 0.87)
macro_rural = macro_rural[macro_rural.geometry.within(pais_vasco_lim.unary_union)]
macro_rural = macro_rural[macro_rural.geometry.apply(clasificar_hex)]
macro_rural['tipo'] = 'macro_rural'
macro_rural['geometry'] = macro_rural.geometry.apply(lambda g: crear_hexagono(g, 14770))

# --- Generar microceldas
micro_urbana = generar_micro_urbanas_personalizado()

# ⚠️ NO ELIMINES microceldas que se solapen con rurales
# macro_union = unary_union(macro_rural.geometry.tolist())
# micro_urbana = micro_urbana[~micro_urbana.geometry.apply(lambda g: prep(macro_union).contains(g))]

# --- Combinar capas finales ---
gdf_cobertura = pd.concat([macro_rural, micro_urbana], ignore_index=True)


# Crear geometría de cobertura actual
cobertura_total = unary_union(gdf_cobertura.geometry)

# Calcular diferencia entre el País Vasco y la cobertura actual
huecos = pais_vasco_lim.unary_union.difference(cobertura_total)

# Convertir a lista de polígonos válidos
if isinstance(huecos, (Polygon, MultiPolygon)):
    huecos = [huecos] if isinstance(huecos, Polygon) else list(huecos.geoms)
elif isinstance(huecos, GeometryCollection):
    huecos = [g for g in huecos.geoms if isinstance(g, (Polygon, MultiPolygon))]
else:
    huecos = []

# Crear hexágonos en las zonas vacías
nuevos_hex_centers = []
dx = np.sqrt(3) * 14770 * 0.95
dy = 1.5 * 14770 * 0.95
minx, miny, maxx, maxy = pais_vasco_lim.total_bounds
y = miny
row = 0
while y < maxy + dy:
    x_offset = 0 if row % 2 == 0 else dx / 2
    x = minx + x_offset
    while x < maxx + dx:
        p = Point(x, y)
        hexagono = crear_hexagono(p, 14770)
        if any(h.intersects(hexagono) for h in huecos):
            nuevos_hex_centers.append(p)
        x += dx
    y += dy
    row += 1


hex_huecos = gpd.GeoDataFrame({
    'tipo': 'macro_rural',
    'geometry': [crear_hexagono(p, 14770) for p in nuevos_hex_centers]
}, crs=pais_vasco_lim.crs)

# --- FILTRAR Y AÑADIR HEXÁGONOS DE HUECOS ---

umbral_interseccion_pais_vasco = 0.05  # mínimo 10% de área dentro del País Vasco
umbral_solapamiento_existente = 0.3   # máximo 10% de solapamiento con cobertura ya existente

# Unión del País Vasco y de cobertura actual
union_pais_vasco = pais_vasco_lim.unary_union
union_actual = unary_union(gdf_cobertura.geometry)

# Crear hexágonos solo si:
# - intersectan significativamente con el País Vasco
# - no están ya cubiertos casi por completo
hex_huecos_filtrados = []
for p in nuevos_hex_centers:
    h = crear_hexagono(p, 14770)

    interseccion_pv = h.intersection(union_pais_vasco).area / h.area
    interseccion_existente = h.intersection(union_actual).area / h.area

    if interseccion_pv > umbral_interseccion_pais_vasco and interseccion_existente < (1 - umbral_solapamiento_existente):
        hex_huecos_filtrados.append(h)

# Crear GeoDataFrame final
hex_huecos = gpd.GeoDataFrame({
    'tipo': 'macro_rural',
    'geometry': hex_huecos_filtrados
}, crs=pais_vasco_lim.crs)

# Añadir a la cobertura total
gdf_cobertura = pd.concat([gdf_cobertura, hex_huecos], ignore_index=True)



# --- Conteo por tipo y ciudad ---
print("\nNúmero de celdas por tipo:")
print(gdf_cobertura['tipo'].value_counts())

conteo_micro = Counter(micro_urbana['ciudad'])
print("\nNúmero de microceldas generadas por ciudad:")
for ciudad in sorted(conteo_micro):
    cantidad = conteo_micro[ciudad]
    print(f"{ciudad.title()}: {cantidad}")


# Ver ciudades urbanas que no generaron ninguna celda
ciudades_urbanas = set(poblacion[poblacion['tipo_entorno'] == 'urbano']['ciudad_limpia'])
ciudades_con_celdas = set(micro_urbana['ciudad'])
faltan = ciudades_urbanas - ciudades_con_celdas

print("\nCiudades urbanas que no generaron celdas:")
for ciudad in sorted(faltan):
    print(ciudad.title())
    
print(f"\nNúmero de nuevos hexágonos añadidos por huecos: {len(hex_huecos)}")
# --- Visualization ---
fig, ax = plt.subplots(figsize=(12, 12))

# Plot boundaries
pais_vasco_lim.boundary.plot(ax=ax, color='black', linewidth=0.5, zorder=1)

# Plot macro rural cells (light blue)
gdf_cobertura[gdf_cobertura['tipo'] == 'macro_rural'].plot(
    ax=ax, color='lightblue', edgecolor='blue', alpha=0.35, linewidth=0.4, zorder=2
)

# Plot micro urban cells (red)
micro_urbana.plot(
    ax=ax, color='red', edgecolor='darkred', alpha=0.4, linewidth=0.2, zorder=3
)

# Plot population boundaries
poblacion.boundary.plot(ax=ax, color='black', linewidth=0.5, zorder=4)

# Legend patches
legend_patches = [
    mpatches.Patch(facecolor='lightblue', edgecolor='blue', label='Macro Rural Cell'),
    mpatches.Patch(facecolor='red', edgecolor='darkred', label='Micro Urban Cell')
]

# Title and legend
plt.title("5G Coverage in the Basque Country: Macro Rural and Micro Urban Cells", fontsize=14)
plt.legend(handles=legend_patches, loc='lower left', fontsize=10)
plt.axis('off')

plt.show()