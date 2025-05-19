import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
import os

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
    print("Columnas disponibles en UPAs:", upas_gdf.columns.tolist())
    
    # Cargar archivo de Universidades
    servicios_path = Path("Infraestructura_Educación_Salud")
    universidades_file = servicios_path / "Universidades.shp"
    
    if not universidades_file.exists():
        raise FileNotFoundError("No se encontró el archivo Universidades.shp")
    
    universidades_gdf = gpd.read_file(universidades_file)
    print("Columnas disponibles en Universidades:", universidades_gdf.columns.tolist())
    
    # Asegurar que ambos GeoDataFrames estén en el mismo CRS (WGS84)
    target_crs = "EPSG:4326"
    upas_gdf = upas_gdf.to_crs(target_crs)
    universidades_gdf = universidades_gdf.to_crs(target_crs)
    
    return upas_gdf, universidades_gdf

def calcular_distancias_por_municipio(upas_gdf, universidades_gdf):
    """
    Calcula las distancias promedio entre UPAs y universidades por municipio.
    """
    # Asegurar que ambos GeoDataFrames estén en el mismo CRS para cálculos de distancia
    # Usamos EPSG:3857 (Web Mercator) para cálculos de distancia en metros
    upas_proj = upas_gdf.to_crs("EPSG:3857")
    universidades_proj = universidades_gdf.to_crs("EPSG:3857")
    
    # Extraer coordenadas de los puntos
    upa_coords = np.array([(geom.x, geom.y) for geom in upas_proj.geometry])
    univ_coords = np.array([(geom.x, geom.y) for geom in universidades_proj.geometry])
    
    # Calcular matriz de distancias
    distancias = cdist(upa_coords, univ_coords)
    
    # Encontrar la distancia mínima para cada UPA
    distancias_min = np.min(distancias, axis=1)
    
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
    upas_gdf, universidades_gdf = cargar_capas_geograficas()
    
    # Calcular distancias por municipio
    print("Calculando distancias promedio por municipio...")
    resultados_df = calcular_distancias_por_municipio(upas_gdf, universidades_gdf)
    
    # Guardar resultados en la carpeta data
    output_file = data_dir / 'distancias_promedio_universidades_por_municipio.csv'
    resultados_df.to_csv(output_file, index=False)
    print(f"Resultados guardados en '{output_file}'")

if __name__ == "__main__":
    main() 