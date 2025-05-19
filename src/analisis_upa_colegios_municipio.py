import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
import os

def cargar_capas_geograficas():
    """
    Carga las capas geográficas de UPAs y el archivo de Colegios.
    """
    # Cargar UPAs
    upas_path = Path("CNA UPAS")
    upas_files = list(upas_path.glob("*.shp"))
    if not upas_files:
        raise FileNotFoundError("No se encontraron archivos .shp en la carpeta CNA UPAS")
    
    upas_gdf = gpd.read_file(upas_files[0])
    print("Columnas disponibles en UPAs:", upas_gdf.columns.tolist())
    
    # Cargar archivo de Colegios
    servicios_path = Path("Infraestructura_Educación_Salud")
    colegios_file = servicios_path / "Colegios.shp"
    
    if not colegios_file.exists():
        raise FileNotFoundError("No se encontró el archivo Colegios.shp")
    
    colegios_gdf = gpd.read_file(colegios_file)
    print("Columnas disponibles en Colegios:", colegios_gdf.columns.tolist())
    
    # Asegurar que ambos GeoDataFrames estén en el mismo CRS (WGS84)
    target_crs = "EPSG:4326"
    upas_gdf = upas_gdf.to_crs(target_crs)
    colegios_gdf = colegios_gdf.to_crs(target_crs)
    
    return upas_gdf, colegios_gdf

def calcular_distancias_por_municipio(upas_gdf, colegios_gdf, batch_size=100):
    """
    Calcula las distancias promedio entre UPAs y colegios por municipio.
    Procesa los colegios en lotes para evitar problemas de memoria.
    """
    # Asegurar que ambos GeoDataFrames estén en el mismo CRS para cálculos de distancia
    # Usamos EPSG:3857 (Web Mercator) para cálculos de distancia en metros
    upas_proj = upas_gdf.to_crs("EPSG:3857")
    colegios_proj = colegios_gdf.to_crs("EPSG:3857")
    
    # Extraer coordenadas de los puntos de UPAs
    upa_coords = np.array([(geom.x, geom.y) for geom in upas_proj.geometry])
    
    # Inicializar array para almacenar las distancias mínimas
    distancias_min = np.full(len(upa_coords), np.inf)
    
    # Procesar colegios en lotes
    total_colegios = len(colegios_proj)
    for i in range(0, total_colegios, batch_size):
        print(f"Procesando colegios {i+1} a {min(i+batch_size, total_colegios)} de {total_colegios}...")
        
        # Obtener lote de colegios
        batch_colegios = colegios_proj.iloc[i:i+batch_size]
        colegio_coords = np.array([(geom.x, geom.y) for geom in batch_colegios.geometry])
        
        # Calcular distancias para este lote
        distancias_batch = cdist(upa_coords, colegio_coords)
        
        # Actualizar distancias mínimas
        distancias_min = np.minimum(distancias_min, np.min(distancias_batch, axis=1))
    
    # Crear DataFrame con resultados
    resultados = pd.DataFrame({
        'municipio_id': upas_gdf['Mpio'],
        'distancia_promedio_servicio_social': distancias_min
    })
    
    # Agrupar por municipio y calcular el promedio
    resultados_agrupados = resultados.groupby('municipio_id')['distancia_promedio_servicio_social'].mean().reset_index()
    
    return resultados_agrupados

def main():
    # Crear directorio de datos si no existe
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Cargar datos
    print("Cargando capas geográficas...")
    upas_gdf, colegios_gdf = cargar_capas_geograficas()
    
    # Calcular distancias por municipio
    print("Calculando distancias promedio por municipio...")
    resultados_df = calcular_distancias_por_municipio(upas_gdf, colegios_gdf)
    
    # Guardar resultados en la carpeta data
    output_file = data_dir / 'distancias_promedio_colegios_por_municipio.csv'
    resultados_df.to_csv(output_file, index=False)
    print(f"Resultados guardados en '{output_file}'")

if __name__ == "__main__":
    main() 