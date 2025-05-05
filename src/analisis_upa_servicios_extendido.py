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
    Carga las capas geográficas de UPAs y los archivos de servicios sociales.
    """
    # Cargar UPAs
    upas_path = Path("CNA UPAS")
    upas_files = list(upas_path.glob("*.shp"))
    if not upas_files:
        raise FileNotFoundError("No se encontraron archivos .shp en la carpeta CNA UPAS")
    
    upas_gdf = gpd.read_file(upas_files[0])
    
    # Cargar archivos de servicios sociales
    servicios_path = Path("Infraestructura_Educación_Salud")
    
    # Lista de servicios a cargar
    servicios = {
        "Universidades": "Universidades.shp",
        "Colegios": "Colegios.shp",
        "Colegios_rurales": "Colegios rurales_Nucleos de priorizacion.shp",
        "Hospitales": "Hospitales.shp"
    }
    
    servicios_gdfs = {}
    
    for nombre, archivo in servicios.items():
        archivo_path = servicios_path / archivo
        if not archivo_path.exists():
            print(f"Advertencia: No se encontró el archivo {archivo}")
            continue
        
        servicios_gdfs[nombre] = gpd.read_file(archivo_path)
    
    # Asegurar que todos los GeoDataFrames estén en el mismo CRS (WGS84)
    target_crs = "EPSG:4326"
    upas_gdf = upas_gdf.to_crs(target_crs)
    
    for nombre, gdf in servicios_gdfs.items():
        servicios_gdfs[nombre] = gdf.to_crs(target_crs)
    
    return upas_gdf, servicios_gdfs

def calcular_distancias_por_lotes(upas_gdf, servicios_gdfs, servicio_nombre, batch_size=1000):
    """
    Calcula las distancias entre cada UPA y el servicio social más cercano en lotes para optimizar memoria.
    
    Args:
        upas_gdf: GeoDataFrame con las UPAs
        servicios_gdfs: Diccionario con los GeoDataFrames de servicios
        servicio_nombre: Nombre del servicio a analizar
        batch_size: Tamaño del lote para procesar
        
    Returns:
        DataFrame con los resultados de las distancias
    """
    if servicio_nombre not in servicios_gdfs:
        raise ValueError(f"El servicio {servicio_nombre} no está disponible")
    
    # Asegurar que ambos GeoDataFrames estén en el mismo CRS para cálculos de distancia
    upas_proj = upas_gdf.to_crs("EPSG:3857")
    servicio_proj = servicios_gdfs[servicio_nombre].to_crs("EPSG:3857")
    
    # Extraer coordenadas de los puntos de servicio una sola vez
    servicio_coords = np.array([(geom.x, geom.y) for geom in servicio_proj.geometry])
    
    # Obtener nombres de columnas para ID y nombre del servicio
    id_col = None
    nombre_col = None
    
    for col in servicios_gdfs[servicio_nombre].columns:
        if col.upper() in ['ID', 'CODIGO', 'COD', 'CÓDIGO']:
            id_col = col
        elif col.upper() in ['NOMBRE', 'NAME', 'NOM', 'NOMBRE_ESTABLECIMIENTO']:
            nombre_col = col
    
    if id_col is None:
        id_col = servicios_gdfs[servicio_nombre].columns[0]
    
    if nombre_col is None:
        nombre_col = servicios_gdfs[servicio_nombre].columns[1] if len(servicios_gdfs[servicio_nombre].columns) > 1 else id_col
    
    # Procesar UPAs en lotes
    resultados = []
    total_upas = len(upas_proj)
    
    for i in range(0, total_upas, batch_size):
        print(f"Procesando lote {i//batch_size + 1} de {(total_upas + batch_size - 1)//batch_size}")
        
        # Obtener el lote actual de UPAs
        batch_end = min(i + batch_size, total_upas)
        batch_upas = upas_proj.iloc[i:batch_end]
        
        # Extraer coordenadas del lote actual
        upa_coords = np.array([(geom.x, geom.y) for geom in batch_upas.geometry])
        
        # Calcular distancias para el lote actual
        distancias = cdist(upa_coords, servicio_coords)
        idx_min = np.argmin(distancias, axis=1)
        distancias_min = np.min(distancias, axis=1)
        
        # Crear DataFrame para el lote actual
        batch_resultados = pd.DataFrame({
            'UPA_ID': batch_upas.get('ID', range(i, batch_end)),
            f'{servicio_nombre}_ID': servicios_gdfs[servicio_nombre].iloc[idx_min][id_col],
            f'Nombre_{servicio_nombre}': servicios_gdfs[servicio_nombre].iloc[idx_min][nombre_col],
            'Distancia': distancias_min
        })
        
        resultados.append(batch_resultados)
        
        # Liberar memoria
        del distancias, idx_min, distancias_min, batch_resultados
        gc.collect()
    
    # Combinar todos los resultados
    return pd.concat(resultados, ignore_index=True)

def crear_visualizaciones(upas_gdf, servicios_gdfs, resultados_df, servicio_nombre):
    """
    Crea las visualizaciones para un servicio específico.
    
    Args:
        upas_gdf: GeoDataFrame con las UPAs
        servicios_gdfs: Diccionario con los GeoDataFrames de servicios
        resultados_df: DataFrame con los resultados de las distancias
        servicio_nombre: Nombre del servicio a visualizar
    """
    # 1. Mapa de dispersión
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plotear UPAs
    upas_gdf.plot(ax=ax, color='red', markersize=50, alpha=0.6, label='UPAs')
    
    # Plotear servicio
    servicios_gdfs[servicio_nombre].plot(
        ax=ax, 
        color='blue', 
        markersize=100, 
        alpha=0.6, 
        label=servicio_nombre
    )
    
    # Agregar fondo de mapa
    try:
        ctx.add_basemap(ax, crs=upas_gdf.crs.to_string())
    except Exception as e:
        print(f"Advertencia: No se pudo agregar el mapa base: {e}")
    
    plt.title(f'UPAs y {servicio_nombre}')
    plt.legend()
    plt.savefig(f'images/mapa_dispersion_{servicio_nombre.lower()}.png')
    plt.close()
    
    # 2. Mapa de calor de distancias
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Crear mapa de calor usando las distancias
    upas_gdf['distancia_servicio'] = resultados_df['Distancia']
    upas_gdf.plot(
        column='distancia_servicio',
        ax=ax,
        legend=True,
        legend_kwds={'label': f'Distancia a {servicio_nombre} más cercano (metros)'},
        cmap='YlOrRd'
    )
    
    # Agregar fondo de mapa
    try:
        ctx.add_basemap(ax, crs=upas_gdf.crs.to_string())
    except Exception as e:
        print(f"Advertencia: No se pudo agregar el mapa base: {e}")
    
    plt.title(f'Mapa de Calor de Distancias a {servicio_nombre}')
    plt.savefig(f'images/mapa_calor_{servicio_nombre.lower()}.png')
    plt.close()

def main():
    # Crear directorio de imágenes si no existe
    os.makedirs('images', exist_ok=True)
    
    # Cargar datos
    print("Cargando capas geográficas...")
    upas_gdf, servicios_gdfs = cargar_capas_geograficas()
    
    # Lista de servicios a analizar
    servicios = ["Colegios", "Colegios_rurales", "Hospitales"]
    
    # Procesar cada servicio
    for servicio in servicios:
        if servicio not in servicios_gdfs:
            print(f"Advertencia: El servicio {servicio} no está disponible. Continuando con el siguiente...")
            continue
        
        print(f"Calculando distancias a {servicio}...")
        resultados_df = calcular_distancias_por_lotes(upas_gdf, servicios_gdfs, servicio, batch_size=500)
        
        # Guardar resultados
        resultados_df.to_csv(f'resultados_distancias_{servicio.lower()}.csv', index=False)
        print(f"Resultados guardados en 'resultados_distancias_{servicio.lower()}.csv'")
        
        # Crear visualizaciones
        print(f"Generando visualizaciones para {servicio}...")
        crear_visualizaciones(upas_gdf, servicios_gdfs, resultados_df, servicio)
        print(f"Visualizaciones para {servicio} guardadas en la carpeta 'images'")
        
        # Liberar memoria
        del resultados_df
        gc.collect()

if __name__ == "__main__":
    main() 