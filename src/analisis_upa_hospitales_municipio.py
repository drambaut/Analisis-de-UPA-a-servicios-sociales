import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
import os

def cargar_capas_geograficas():
    """
    Carga las capas geográficas de UPAs y el archivo de Hospitales.
    """
    # Cargar UPAs
    upas_path = Path("CNA UPAS")
    upas_files = list(upas_path.glob("*.shp"))
    if not upas_files:
        raise FileNotFoundError("No se encontraron archivos .shp en la carpeta CNA UPAS")
    
    upas_gdf = gpd.read_file(upas_files[0])
    print("Columnas disponibles en UPAs:", upas_gdf.columns.tolist())
    
    # Cargar archivo de Hospitales
    servicios_path = Path("Infraestructura_Educación_Salud")
    hospitales_file = servicios_path / "Hospitales.shp"
    
    if not hospitales_file.exists():
        raise FileNotFoundError("No se encontró el archivo Hospitales.shp")
    
    hospitales_gdf = gpd.read_file(hospitales_file)
    print("Columnas disponibles en Hospitales:", hospitales_gdf.columns.tolist())
    
    # Asegurar que ambos GeoDataFrames estén en el mismo CRS (WGS84)
    target_crs = "EPSG:4326"
    upas_gdf = upas_gdf.to_crs(target_crs)
    hospitales_gdf = hospitales_gdf.to_crs(target_crs)
    
    return upas_gdf, hospitales_gdf

def calcular_distancias_por_municipio(upas_gdf, hospitales_gdf, batch_size=100):
    """
    Calcula las distancias promedio entre UPAs y hospitales por municipio.
    Procesa los hospitales en lotes para evitar problemas de memoria.
    """
    # Asegurar que ambos GeoDataFrames estén en el mismo CRS para cálculos de distancia
    # Usamos EPSG:3857 (Web Mercator) para cálculos de distancia en metros
    upas_proj = upas_gdf.to_crs("EPSG:3857")
    hospitales_proj = hospitales_gdf.to_crs("EPSG:3857")
    
    # Extraer coordenadas de los puntos de UPAs
    upa_coords = np.array([(geom.x, geom.y) for geom in upas_proj.geometry])
    
    # Inicializar array para almacenar las distancias mínimas
    distancias_min = np.full(len(upa_coords), np.inf)
    
    # Procesar hospitales en lotes
    total_hospitales = len(hospitales_proj)
    for i in range(0, total_hospitales, batch_size):
        print(f"Procesando hospitales {i+1} a {min(i+batch_size, total_hospitales)} de {total_hospitales}...")
        
        # Obtener lote de hospitales
        batch_hospitales = hospitales_proj.iloc[i:i+batch_size]
        hosp_coords = np.array([(geom.x, geom.y) for geom in batch_hospitales.geometry])
        
        # Calcular distancias para este lote
        distancias_batch = cdist(upa_coords, hosp_coords)
        
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
    upas_gdf, hospitales_gdf = cargar_capas_geograficas()
    
    # Calcular distancias por municipio
    print("Calculando distancias promedio por municipio...")
    resultados_df = calcular_distancias_por_municipio(upas_gdf, hospitales_gdf)
    
    # Guardar resultados en la carpeta data
    output_file = data_dir / 'distancias_promedio_hospitales_por_municipio.csv'
    resultados_df.to_csv(output_file, index=False)
    print(f"Resultados guardados en '{output_file}'")

if __name__ == "__main__":
    main() 