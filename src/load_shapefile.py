import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import os
from pathlib import Path

def load_and_visualize_shapefile():
    try:
        # Get the current working directory
        workspace_dir = os.getcwd()
        shapefile_path = os.path.join(workspace_dir, "CNA UPAS", "CNA_Geo.shp")
        
        print(f"Current working directory: {workspace_dir}")
        print(f"Attempting to load shapefile from: {shapefile_path}")
        
        # Check if file exists
        if not os.path.exists(shapefile_path):
            print(f"Error: Shapefile not found at {shapefile_path}")
            return None
            
        # Load the shapefile
        upa = gpd.read_file(shapefile_path)
        
        # Basic information about the data
        print("Shapefile Information:")
        print(f"Number of features: {len(upa)}")
        print(f"Columns: {upa.columns.tolist()}")
        print(f"CRS: {upa.crs}")
        
        # Create a basic plot
        fig, ax = plt.subplots(figsize=(12, 8))
        upa.plot(ax=ax, alpha=0.5, edgecolor='k')
        
        try:
            # Add basemap
            ctx.add_basemap(ax, crs=upa.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print(f"Warning: Could not add basemap: {str(e)}")
        
        # Customize the plot
        plt.title('Visualización de UPAS')
        plt.axis('off')
        
        # Create images directory if it doesn't exist
        os.makedirs('images', exist_ok=True)
        
        # Save the plot
        plt.savefig('images/upa_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return upa
    except Exception as e:
        print(f"Error loading shapefile: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    upa_data = load_and_visualize_shapefile() 