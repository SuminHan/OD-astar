from settings import *
import geopandas as gpd

class Map:
    def __init__(self):
        self.lte_gdf = gpd.read_file(f'osm_data/{MAP_LABEL}_lte_cell.geojson')
        self.mx1, self.my1, self.mx2, self.my2 = self.lte_gdf.total_bounds
        self.map_width = self.mx2 - self.mx1
        self.map_height = self.my2 - self.my1
        self.grid_width = self.lte_gdf['i'].max()+1
        self.grid_height = self.lte_gdf['j'].max()+1

        self.building_gdf = gpd.read_file(f'osm_data/{MAP_LABEL}_building_official.geojson')
        self.edge_gdf = gpd.read_file(f'osm_data/{MAP_LABEL}_edge.geojson')#, dtype={'u':str, 'v':str})
        self.node_gdf = gpd.read_file(f'osm_data/{MAP_LABEL}_node.geojson')#, dtype={'osmid': str})
        print(f'loading map {MAP_LABEL} complete.')

    def change_coord(self, x, y):
        scale_x = SCREEN_WIDTH / self.map_width
        scale_y = SCREEN_HEIGHT / self.map_height
        rx = scale_x * (x - self.mx1)
        ry = SCREEN_HEIGHT - scale_y * (y - self.my1)
        return rx, ry
    
    def get_buildings(self):
        vertice_list = []
        for geo in self.building_gdf.geometry:
            xx, yy = geo.exterior.coords.xy
            vertices = [self.change_coord(x, y) for x, y in zip(xx, yy)]
            vertice_list.append(vertices[:-1])
        return vertice_list

    def get_grids(self):
        vertice_list = []
        for geo in self.lte_gdf.geometry:
            xx, yy = geo.exterior.coords.xy
            vertices = [self.change_coord(x, y) for x, y in zip(xx, yy)]
            vertice_list.append(vertices[:-1])
        return vertice_list

    def get_roads(self):
        vertice_list = []
        for geo in self.edge_gdf.geometry:
            xx, yy = geo.coords.xy
            vertices = [self.change_coord(x, y) for x, y in zip(xx, yy)]
            vertice_list.append(vertices)
            # print(vertices)
        return vertice_list
        
    def get_nodes(self):
        vertice_list = []
        for geo in self.node_gdf.geometry:
            xx, yy = geo.coords.xy
            assert len(xx) == len(yy) == 1
            vertice_list.append(self.change_coord(xx[0], yy[0]))
            # print(vertices)
        return vertice_list

# map = Map()
#print(map.node_gdf, map.node_gdf['osmid'])
# print(map.edge_gdf, map.edge_gdf['u'], map.edge_gdf['v'])
# print(map.get_buildings())
# print(map.get_roads())
# print(map.get_nodes())




