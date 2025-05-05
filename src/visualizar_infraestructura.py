import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from pathlib import Path

def cargar_shape(shape_path):
    """Carga un archivo shape y retorna un GeoDataFrame"""
    return gpd.read_file(shape_path)

def visualizar_shape(gdf, titulo, color='red', alpha=0.6):
    """Visualiza un GeoDataFrame en un mapa"""
    # Crear figura y eje
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Proyectar a Web Mercator para usar con contextily
    gdf_web = gdf.to_crs(epsg=3857)
    
    # Graficar el shape
    gdf_web.plot(ax=ax, color=color, alpha=alpha)
    
    # Agregar mapa base
    ctx.add_basemap(ax)
    
    # Configurar título y eliminar ejes
    ax.set_title(titulo, fontsize=16, pad=20)
    ax.set_axis_off()
    
    return fig

def main():
    # Definir la ruta base donde están los shapes
    base_path = Path("Infraestructura_Educación_Salud")
    
    # Diccionario con los shapes a visualizar y sus colores
    shapes = {
        "Universidades": {"file": "Universidades.shp", "color": "blue"},
        "Colegios": {"file": "Colegios.shp", "color": "green"},
        "Colegios_rurales": {"file": "Colegios rurales_Nucleos de priorizacion.shp", "color": "orange"},
        "Hospitales": {"file": "Hospitales.shp", "color": "red"}
    }
    
    # Visualizar cada shape por separado
    for nombre, info in shapes.items():
        try:
            # Cargar el shape
            gdf = cargar_shape(base_path / info["file"])
            
            # Visualizar
            fig = visualizar_shape(gdf, nombre, color=info["color"])
            
            # Guardar la figura
            plt.savefig(f"images/{nombre.lower()}_mapa.png", 
                       bbox_inches='tight', 
                       dpi=300)
            plt.close()
            
            print(f"Mapa de {nombre} generado exitosamente")
            
        except Exception as e:
            print(f"Error al procesar {nombre}: {str(e)}")

if __name__ == "__main__":
    main() 