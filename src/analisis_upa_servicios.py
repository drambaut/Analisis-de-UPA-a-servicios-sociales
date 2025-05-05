import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from pathlib import Path
import os
from scipy.spatial.distance import cdist
import numpy as np

def cargar_capas_geograficas():
    """
    Carga las capas geográficas de UPAs y el archivo de Universidades.
    """
    # Cargar UPAs
    upas_path = Path("CNA UPAS")
    upas_files = list(upas_path.glob("*.shp"))
    if not upas_files:
        raise FileNotFoundError("No se encontraron archivos .shp en la carpeta CNA UPAS")
    
    upas_gdf = gpd.read_file(upas_files[0])
    
    # Cargar archivo de Universidades
    servicios_path = Path("Infraestructura_Educación_Salud")
    universidades_file = servicios_path / "Universidades.shp"
    
    if not universidades_file.exists():
        raise FileNotFoundError("No se encontró el archivo Universidades.shp")
    
    universidades_gdf = gpd.read_file(universidades_file)
    
    # Asegurar que ambos GeoDataFrames estén en el mismo CRS (WGS84)
    target_crs = "EPSG:4326"
    upas_gdf = upas_gdf.to_crs(target_crs)
    universidades_gdf = universidades_gdf.to_crs(target_crs)
    
    return upas_gdf, {"Universidades": universidades_gdf}

def calcular_distancias(upas_gdf, servicios_gdfs):
    """
    Calcula las distancias entre cada UPA y las universidades más cercanas de forma vectorizada.
    """
    # Asegurar que ambos GeoDataFrames estén en el mismo CRS para cálculos de distancia
    # Usamos EPSG:3857 (Web Mercator) para cálculos de distancia en metros
    upas_proj = upas_gdf.to_crs("EPSG:3857")
    universidades_proj = servicios_gdfs["Universidades"].to_crs("EPSG:3857")
    
    # Extraer coordenadas de los puntos
    upa_coords = np.array([(geom.x, geom.y) for geom in upas_proj.geometry])
    univ_coords = np.array([(geom.x, geom.y) for geom in universidades_proj.geometry])
    
    # Calcular matriz de distancias
    distancias = cdist(upa_coords, univ_coords)
    
    # Encontrar índices de las universidades más cercanas
    idx_min = np.argmin(distancias, axis=1)
    distancias_min = np.min(distancias, axis=1)
    
    # Crear DataFrame de resultados
    resultados = pd.DataFrame({
        'UPA_ID': upas_gdf.get('ID', range(len(upas_gdf))),
        'Universidad_ID': servicios_gdfs["Universidades"].iloc[idx_min].get('ID', idx_min),
        'Nombre_Universidad': servicios_gdfs["Universidades"].iloc[idx_min].get('NOMBRE', 'Sin nombre'),
        'Distancia': distancias_min
    })
    
    return resultados

def crear_visualizaciones(upas_gdf, servicios_gdfs, resultados_df):
    """
    Crea las visualizaciones solicitadas.
    """
    # 1. Mapa de dispersión
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plotear UPAs
    upas_gdf.plot(ax=ax, color='red', markersize=50, alpha=0.6, label='UPAs')
    
    # Plotear universidades
    servicios_gdfs["Universidades"].plot(
        ax=ax, 
        color='blue', 
        markersize=100, 
        alpha=0.6, 
        label='Universidades'
    )
    
    # Agregar fondo de mapa
    try:
        ctx.add_basemap(ax, crs=upas_gdf.crs.to_string())
    except Exception as e:
        print(f"Advertencia: No se pudo agregar el mapa base: {e}")
    
    plt.title('UPAs y Universidades')
    plt.legend()
    plt.savefig('images/mapa_dispersion_universidades.png')
    plt.close()
    
    # 2. Mapa de calor de distancias
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Crear mapa de calor usando las distancias
    upas_gdf['distancia_universidad'] = resultados_df['Distancia']
    upas_gdf.plot(
        column='distancia_universidad',
        ax=ax,
        legend=True,
        legend_kwds={'label': 'Distancia a Universidad más cercana (metros)'},
        cmap='YlOrRd'
    )
    
    # Agregar fondo de mapa
    try:
        ctx.add_basemap(ax, crs=upas_gdf.crs.to_string())
    except Exception as e:
        print(f"Advertencia: No se pudo agregar el mapa base: {e}")
    
    plt.title('Mapa de Calor de Distancias a Universidades')
    plt.savefig('images/mapa_calor_universidades.png')
    plt.close()

def main():
    # Crear directorio de imágenes si no existe
    os.makedirs('images', exist_ok=True)
    
    # Cargar datos
    print("Cargando capas geográficas...")
    upas_gdf, servicios_gdfs = cargar_capas_geograficas()
    
    # Calcular distancias
    print("Calculando distancias a universidades...")
    resultados_df = calcular_distancias(upas_gdf, servicios_gdfs)
    
    # Guardar resultados
    resultados_df.to_csv('resultados_distancias_universidades.csv', index=False)
    print("Resultados guardados en 'resultados_distancias_universidades.csv'")
    
    # Crear visualizaciones
    print("Generando visualizaciones...")
    crear_visualizaciones(upas_gdf, servicios_gdfs, resultados_df)
    print("Visualizaciones guardadas en la carpeta 'images'")

if __name__ == "__main__":
    main() 