import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from pathlib import Path
import os
from scipy.spatial.distance import cdist
import numpy as np
import gc

def cargar_capas_geograficas():
    """
    Carga las capas geográficas de UPAs y el archivo de Colegios Rurales.
    """
    # Cargar UPAs
    upas_path = Path("CNA UPAS")
    upas_files = list(upas_path.glob("*.shp"))
    if not upas_files:
        raise FileNotFoundError("No se encontraron archivos .shp en la carpeta CNA UPAS")
    
    upas_gdf = gpd.read_file(upas_files[0])
    print("Columnas disponibles en UPAs:", upas_gdf.columns.tolist())
    
    # Cargar archivo de Colegios Rurales
    servicios_path = Path("Infraestructura_Educación_Salud")
    colegios_file = servicios_path / "Colegios rurales_Nucleos de priorizacion.shp"
    
    if not colegios_file.exists():
        raise FileNotFoundError("No se encontró el archivo de Colegios Rurales")
    
    colegios_gdf = gpd.read_file(colegios_file)
    print("Columnas disponibles en Colegios Rurales:", colegios_gdf.columns.tolist())
    
    # Asegurar que ambos GeoDataFrames estén en el mismo CRS (WGS84)
    target_crs = "EPSG:4326"
    upas_gdf = upas_gdf.to_crs(target_crs)
    colegios_gdf = colegios_gdf.to_crs(target_crs)
    
    return upas_gdf, colegios_gdf

def calcular_distancias_por_lotes(upas_gdf, colegios_gdf, batch_size=100):
    """
    Calcula las distancias entre cada UPA y los colegios rurales más cercanos procesando en lotes.
    """
    # Asegurar que ambos GeoDataFrames estén en el mismo CRS para cálculos de distancia
    upas_proj = upas_gdf.to_crs("EPSG:3857")
    colegios_proj = colegios_gdf.to_crs("EPSG:3857")
    
    # Extraer coordenadas de los puntos de UPAs
    upa_coords = np.array([(geom.x, geom.y) for geom in upas_proj.geometry])
    
    # Inicializar arrays para almacenar resultados
    distancias_min = np.full(len(upa_coords), np.inf)
    colegio_ids = np.full(len(upa_coords), -1)
    colegio_nombres = np.full(len(upa_coords), '', dtype=object)
    
    # Procesar colegios en lotes
    total_colegios = len(colegios_proj)
    for i in range(0, total_colegios, batch_size):
        print(f"Procesando colegios rurales {i+1} a {min(i+batch_size, total_colegios)} de {total_colegios}...")
        
        # Obtener lote de colegios
        batch_colegios = colegios_proj.iloc[i:i+batch_size]
        
        # Extraer coordenadas de los colegios, manejando tanto Point como MultiPoint
        colegio_coords = []
        for geom in batch_colegios.geometry:
            if geom.geom_type == 'Point':
                colegio_coords.append((geom.x, geom.y))
            elif geom.geom_type == 'MultiPoint':
                # Para MultiPoint, tomamos el primer punto
                colegio_coords.append((geom.geoms[0].x, geom.geoms[0].y))
            else:
                print(f"Advertencia: Geometría no soportada: {geom.geom_type}")
                continue
        
        colegio_coords = np.array(colegio_coords)
        
        # Calcular distancias para este lote
        distancias_batch = cdist(upa_coords, colegio_coords)
        
        # Encontrar índices de los colegios más cercanos en este lote
        idx_min_batch = np.argmin(distancias_batch, axis=1)
        distancias_min_batch = np.min(distancias_batch, axis=1)
        
        # Actualizar resultados donde las distancias son menores
        mask = distancias_min_batch < distancias_min
        distancias_min[mask] = distancias_min_batch[mask]
        colegio_ids[mask] = batch_colegios.index[idx_min_batch[mask]]
        colegio_nombres[mask] = batch_colegios.iloc[idx_min_batch[mask]]['Name'].values
        
        # Liberar memoria
        del distancias_batch, idx_min_batch, distancias_min_batch
        gc.collect()
    
    # Crear DataFrame de resultados
    resultados = pd.DataFrame({
        'UPA_ID': upas_gdf.get('ID', range(len(upas_gdf))),
        'Colegio_Rural_ID': colegio_ids,
        'Nombre_Colegio_Rural': colegio_nombres,
        'Distancia': distancias_min
    })
    
    return resultados

def crear_visualizaciones(upas_gdf, colegios_gdf, resultados_df):
    """
    Crea las visualizaciones para colegios rurales.
    """
    # 1. Mapa de dispersión
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plotear UPAs
    upas_gdf.plot(ax=ax, color='red', markersize=50, alpha=0.6, label='UPAs')
    
    # Plotear colegios rurales
    colegios_gdf.plot(
        ax=ax, 
        color='green', 
        markersize=100, 
        alpha=0.6, 
        label='Colegios Rurales'
    )
    
    # Agregar fondo de mapa
    try:
        ctx.add_basemap(ax, crs=upas_gdf.crs.to_string())
    except Exception as e:
        print(f"Advertencia: No se pudo agregar el mapa base: {e}")
    
    plt.title('UPAs y Colegios Rurales')
    plt.legend()
    plt.savefig('images/mapa_dispersion_colegios_rurales.png')
    plt.close()
    
    # 2. Mapa de calor de distancias
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Crear mapa de calor usando las distancias
    upas_gdf['distancia_colegio_rural'] = resultados_df['Distancia']
    upas_gdf.plot(
        column='distancia_colegio_rural',
        ax=ax,
        legend=True,
        legend_kwds={'label': 'Distancia a Colegio Rural más cercano (metros)'},
        cmap='YlOrRd'
    )
    
    # Agregar fondo de mapa
    try:
        ctx.add_basemap(ax, crs=upas_gdf.crs.to_string())
    except Exception as e:
        print(f"Advertencia: No se pudo agregar el mapa base: {e}")
    
    plt.title('Mapa de Calor de Distancias a Colegios Rurales')
    plt.savefig('images/mapa_calor_colegios_rurales.png')
    plt.close()

def main():
    # Crear directorio de imágenes si no existe
    os.makedirs('images', exist_ok=True)
    
    # Cargar datos
    print("Cargando capas geográficas...")
    upas_gdf, colegios_gdf = cargar_capas_geograficas()
    
    # Calcular distancias
    print("Calculando distancias a colegios rurales...")
    resultados_df = calcular_distancias_por_lotes(upas_gdf, colegios_gdf, batch_size=100)
    
    # Guardar resultados
    resultados_df.to_csv('resultados_distancias_colegios_rurales.csv', index=False)
    print("Resultados guardados en 'resultados_distancias_colegios_rurales.csv'")
    
    # Crear visualizaciones
    print("Generando visualizaciones...")
    crear_visualizaciones(upas_gdf, colegios_gdf, resultados_df)
    print("Visualizaciones guardadas en la carpeta 'images'")

if __name__ == "__main__":
    main() 