import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path
import os
from sklearn.neighbors import BallTree
import gc

def cargar_datos():
    """
    Carga las UPAs y todos los servicios sociales disponibles.
    """
    print("Cargando datos...")
    
    # Cargar UPAs
    upas_path = Path("CNA UPAS/CNA_Geo.shp")
    upas_gdf = gpd.read_file(upas_path)
    print(f"UPAs cargadas: {len(upas_gdf)} registros")
    
    # Cargar servicios sociales
    servicios_path = Path("Infraestructura_Educación_Salud")
    servicios = {}
    
    # Colegios regulares
    colegios_file = servicios_path / "Colegios.shp"
    if colegios_file.exists():
        servicios['colegios'] = gpd.read_file(colegios_file)
        print(f"Colegios cargados: {len(servicios['colegios'])} registros")
    
    # Colegios rurales
    colegios_rurales_file = servicios_path / "Colegios rurales_Nucleos de priorizacion.shp"
    if colegios_rurales_file.exists():
        servicios['colegios_rurales'] = gpd.read_file(colegios_rurales_file)
        print(f"Colegios rurales cargados: {len(servicios['colegios_rurales'])} registros")
    
    # Hospitales
    hospitales_file = servicios_path / "Hospitales.shp"
    if hospitales_file.exists():
        servicios['hospitales'] = gpd.read_file(hospitales_file)
        print(f"Hospitales cargados: {len(servicios['hospitales'])} registros")
    
    # Universidades
    universidades_file = servicios_path / "Universidades.shp"
    if universidades_file.exists():
        servicios['universidades'] = gpd.read_file(universidades_file)
        print(f"Universidades cargadas: {len(servicios['universidades'])} registros")
    
    return upas_gdf, servicios

def calcular_distancias_optimizado(upas_gdf, servicios_gdf, nombre_servicio, batch_size=10000):
    """
    Calcula las distancias mínimas usando BallTree para mayor eficiencia.
    """
    print(f"Calculando distancias a {nombre_servicio}...")
    
    # Convertir a CRS proyectado para cálculos de distancia en metros
    upas_proj = upas_gdf.to_crs("EPSG:3857")
    servicios_proj = servicios_gdf.to_crs("EPSG:3857")
    
    # Extraer coordenadas de servicios (manejar tanto Point como MultiPoint)
    servicio_coords = []
    for geom in servicios_proj.geometry:
        if geom.geom_type == 'Point':
            servicio_coords.append((geom.x, geom.y))
        elif geom.geom_type == 'MultiPoint':
            # Para MultiPoint, tomar el primer punto
            servicio_coords.append((geom.geoms[0].x, geom.geoms[0].y))
        else:
            # Para otros tipos de geometría, usar el centroide
            centroid = geom.centroid
            servicio_coords.append((centroid.x, centroid.y))
    
    servicio_coords = np.array(servicio_coords)
    
    # Crear BallTree para búsqueda eficiente
    tree = BallTree(servicio_coords, metric='euclidean')
    
    # Procesar UPAs en lotes para ahorrar memoria
    distancias_minimas = []
    total_upas = len(upas_proj)
    
    for i in range(0, total_upas, batch_size):
        end_idx = min(i + batch_size, total_upas)
        batch_upas = upas_proj.iloc[i:end_idx]
        
        # Extraer coordenadas del lote (manejar tanto Point como MultiPoint)
        upa_coords_batch = []
        for geom in batch_upas.geometry:
            if geom.geom_type == 'Point':
                upa_coords_batch.append((geom.x, geom.y))
            elif geom.geom_type == 'MultiPoint':
                # Para MultiPoint, tomar el primer punto
                upa_coords_batch.append((geom.geoms[0].x, geom.geoms[0].y))
            else:
                # Para otros tipos de geometría, usar el centroide
                centroid = geom.centroid
                upa_coords_batch.append((centroid.x, centroid.y))
        
        upa_coords_batch = np.array(upa_coords_batch)
        
        # Calcular distancias mínimas para el lote
        distancias_batch, _ = tree.query(upa_coords_batch, k=1)
        distancias_minimas.extend(distancias_batch.flatten())
        
        # Mostrar progreso
        if (i // batch_size) % 10 == 0:
            print(f"  Procesado {end_idx}/{total_upas} UPAs ({(end_idx/total_upas)*100:.1f}%)")
        
        # Liberar memoria
        del batch_upas, upa_coords_batch, distancias_batch
        gc.collect()
    
    return np.array(distancias_minimas)

def crear_tabla_distancias():
    """
    Crea la tabla principal con todas las distancias por UPA.
    """
    # Cargar datos
    upas_gdf, servicios = cargar_datos()
    
    # Crear DataFrame base con información de UPAs
    tabla_distancias = pd.DataFrame({
        'municipio_id': upas_gdf['Mpio'],
        'departamento_id': upas_gdf['depto'],
        'upa_id': upas_gdf['ID_COMPLET'],
        'x_coord': upas_gdf['X_GEO'],
        'y_coord': upas_gdf['Y_GEO']
    })
    
    # Calcular distancias para cada servicio
    if 'colegios' in servicios:
        dist_colegios = calcular_distancias_optimizado(upas_gdf, servicios['colegios'], 'colegios')
        tabla_distancias['distancia_colegios_metros'] = dist_colegios
        del servicios['colegios']
        gc.collect()
    
    if 'colegios_rurales' in servicios:
        dist_colegios_rurales = calcular_distancias_optimizado(upas_gdf, servicios['colegios_rurales'], 'colegios rurales')
        tabla_distancias['distancia_colegios_rurales_metros'] = dist_colegios_rurales
        del servicios['colegios_rurales']
        gc.collect()
    
    if 'hospitales' in servicios:
        dist_hospitales = calcular_distancias_optimizado(upas_gdf, servicios['hospitales'], 'hospitales')
        tabla_distancias['distancia_hospitales_metros'] = dist_hospitales
        del servicios['hospitales']
        gc.collect()
    
    if 'universidades' in servicios:
        dist_universidades = calcular_distancias_optimizado(upas_gdf, servicios['universidades'], 'universidades')
        tabla_distancias['distancia_universidades_metros'] = dist_universidades
        del servicios['universidades']
        gc.collect()
    
    return tabla_distancias

def guardar_resultados(tabla_distancias):
    """
    Guarda los resultados en diferentes formatos.
    """
    # Crear directorio de datos si no existe
    os.makedirs('data', exist_ok=True)
    
    # Guardar como CSV
    archivo_csv = 'data/distancias_upa_servicios_sociales.csv'
    tabla_distancias.to_csv(archivo_csv, index=False)
    print(f"Resultados guardados en: {archivo_csv}")
    
    # Mostrar estadísticas básicas
    print("\nEstadísticas de distancias (en metros):")
    columnas_distancias = [col for col in tabla_distancias.columns if 'distancia' in col]
    
    for col in columnas_distancias:
        print(f"\n{col}:")
        print(f"  Mínima: {tabla_distancias[col].min():.2f}")
        print(f"  Máxima: {tabla_distancias[col].max():.2f}")
        print(f"  Promedio: {tabla_distancias[col].mean():.2f}")
        print(f"  Mediana: {tabla_distancias[col].median():.2f}")
    
    # Mostrar resumen por municipio
    print(f"\nResumen por municipio:")
    print(f"Total de municipios: {tabla_distancias['municipio_id'].nunique()}")
    print(f"Total de UPAs: {len(tabla_distancias)}")
    
    return archivo_csv

def main():
    """
    Función principal que ejecuta todo el análisis.
    """
    print("=== CÁLCULO DE DISTANCIAS UPA-SERVICIOS SOCIALES (OPTIMIZADO) ===")
    
    # Crear tabla de distancias
    tabla_distancias = crear_tabla_distancias()
    
    # Guardar resultados
    archivo_resultado = guardar_resultados(tabla_distancias)
    
    # Mostrar muestra de los datos
    print(f"\nMuestra de los datos generados:")
    print(tabla_distancias.head(10))
    
    print(f"\n¡Análisis completado! Los datos están en: {archivo_resultado}")

if __name__ == "__main__":
    main() 