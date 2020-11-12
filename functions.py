import geopandas as gpd
import folium
import os

from IPython.display import display, HTML, display_html


def display_side_by_side(*args):
    """
    Create side by side display function for dataframes
    """
    html_str=''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table', 'table style="display:inline"'), raw=True)
    
def create_map_from_shapes(shapefile_path, Avrg_sp_hood, middlepoint=[42.05, -93.64], zoom=12):
    """
    Plots the shapefiles on a map and adds a fill color based on average sale price
    """
    # Import the shapefiles from the shapefile directory
    files = [f for f in os.listdir(shapefile_path) if os.path.isfile(os.path.join(shapefile_path, f))]
    files = [x for x in files if '.shp' in x]
    names = [x.split("_",1)[0] for x in files]
    
    # Read the shapes with geopandas
    hood = gpd.read_file(os.path.join(shapefile_path, files[0]))
    for shape in range(1, len(files)):
        hood = hood.append(gpd.read_file(os.path.join(shapefile_path, files[shape])))
        
    # Format geo df and add sales price
    hood['Neighborhood'] = names
    hood = hood.merge(Avrg_sp_hood, on='Neighborhood')
    
    # Plot actual map
    map_ames = folium.Map(middlepoint, zoom_start=zoom)
    map_ames.choropleth(geo_data = hood.to_json(), 
                        data = hood,
                        columns = ['Neighborhood', 'SalePrice'], 
                        key_on = 'feature.properties.{}'.format('Neighborhood'), 
                        fill_color = 'RdBu_r', 
                        fill_opacity = 0.6)
    
    return map_ames



