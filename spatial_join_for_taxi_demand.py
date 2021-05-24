#from qgis.core import *
#from qgis.utils import *
#import os, sys
#
#QgsApplication.setPrefixPath("C:/OSGeo4W64/apps/qgis", True)
#app = QApplication([], True)
#QgsApplication.initQgis()
#
#sys.path.append('C:/OSGEO4~1/apps/qgis/python/plugins')
#from processing.core.Processing import Processing
#Processing.initialize()
#from processing.tools import *
#
#layer1 = "path/to/point_shapefile.shp"
#layer2 = "path/to/polygon_shapefile.shp"
#result = "path/to/output_shapefile.shp"
#
#general.runalg("qgis:joinattributesbylocation", layer1, layer2, u'intersects', 0, 0, '', 1, result)

#shp_uri = 'E:/qgis_python/k=300.shp'
#shp =  QgsVectorLayer(shp_uri, 'k=300', 'ogr')
#QgsProject.instance().addMapLayer(shp)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181101.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181101', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181102.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181102', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181103.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181103', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181104.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181104', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181105.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181105', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181106.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181106', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181107.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181107', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181108.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181108', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181109.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181109', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181110.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181110', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181111.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181111', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181112.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181112', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181113.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181113', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181114.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181114', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181115.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181115', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181116.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181116', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181117.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181117', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181118.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181118', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181119.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181119', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181120.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181120', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181121.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181121', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181122.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181122', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181123.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181123', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181124.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181124', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181125.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181125', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181126.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181126', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181127.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181127', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181128.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181128', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181129.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181129', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
#csv_uri = 'file:///E:/qgis_python/short_link/link_20181130.csv?delimiter=,&xField=V10&yField=V11&crs=epsg:4326'
#csv = QgsVectorLayer(csv_uri, 'link_20181130', 'delimitedtext')
#QgsProject.instance().addMapLayer(csv)
#
# Neighborhood
from qgis.utils import iface
from PyQt5.QtCore import QVariant

_NAME_FIELD = 'cluster2'
_NEW_NEIGHBORS_FIELD = 'NEIGHBORS'

layer = iface.activeLayer()

layer.startEditing()
layer.dataProvider().addAttributes(
        [QgsField(_NEW_NEIGHBORS_FIELD, QVariant.String)]
        )
layer.updateFields()

# Create a dictionary of all features
feature_dict = {f.id(): f for f in layer.getFeatures()}

# Loop through all features and find features that touch each feature
for f in feature_dict.values():
    geom = f.geometry()
    # Find all features that intersect the bounding box of the current feature.
    # We use spatial index to find the features intersecting the bounding box
    # of the current feature. This will narrow down the features that we need
    # to check neighboring features.
    intersecting_ids = index.intersects(geom.boundingBox())
    # Initalize neighbors list and sum
    neighbors = []
    for intersecting_id in intersecting_ids:
        # Look up the feature from the dictionary
        intersecting_f = feature_dict[intersecting_id]

        # For our purpose we consider a feature as 'neighbor' if it touches or
        # intersects a feature. We use the 'disjoint' predicate to satisfy
        # these conditions. So if a feature is not disjoint, it is a neighbor.
        if (f != intersecting_f and not intersecting_f.geometry().disjoint(geom)):
            neighbors.append(intersecting_f[_NAME_FIELD])
    f[_NEW_NEIGHBORS_FIELD] = ','.join(neighbors)
    # Update the layer with new attribute values.
    layer.updateFeature(f)

layer.commitChanges()
print('Processing complete.')




























