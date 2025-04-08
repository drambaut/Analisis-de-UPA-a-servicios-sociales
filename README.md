# Análisis de UPAS a Servicios Sociales

Este proyecto contiene el análisis de las Unidades de Producción Agropecuaria (UPAS) y su relación con servicios sociales.

## Estructura del Proyecto

```
.
├── data/               # Datos del proyecto
├── notebooks/          # Jupyter notebooks para análisis
├── src/                # Código fuente Python
└── CNA UPAS/          # Shapefiles originales
```

## Configuración del Entorno

1. Crear un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Para visualizar el shapefile:
```bash
python src/load_shapefile.py
```

2. Para análisis interactivo, abrir el notebook:
```bash
jupyter notebook notebooks/upa_analysis.ipynb
```

## Dependencias Principales

- pandas
- geopandas
- matplotlib
- jupyter
- numpy
- folium
- contextily