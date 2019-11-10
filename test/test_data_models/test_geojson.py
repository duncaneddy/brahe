# Test Imports
from pytest import approx

# Modules Under Test
import brahe.data_models as bdm

def test_geojson_point():
    point = {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [102.0, 0.5]
      },
      "properties": {
        "prop0": "value0"
      }
    }

    geopoint = bdm.GeoJSONObject(**point)

    assert geopoint.type == 'Feature'
    assert geopoint.geometry.type == 'Point'
    assert len(geopoint.geometry.coordinates) == 2
    assert geopoint.properties['prop0'] == 'value0'


    import logging
    logging.info(geopoint.dict())

def test_geojson_linestring():
    linestring = {
      "type": "Feature",
      "geometry": {
        "type": "LineString",
        "coordinates": [
          [102.0, 0.0], [103.0, 1.0], [104.0, 0.0], [105.0, 1.0]
        ]
      },
      "properties": {
        "prop0": "value0",
        "prop1": 0.0
      }
    }

    geolinestring = bdm.GeoJSONObject(**linestring)

    assert geolinestring.type == 'Feature'
    assert geolinestring.geometry.type == 'LineString'
    assert len(geolinestring.geometry.coordinates) == 4
    assert len(geolinestring.geometry.coordinates[0]) == 2
    assert geolinestring.properties['prop0'] == 'value0'

def test_geojson_polygon():
    polygon = {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [100.0, 0.0], [101.0, 0.0], [101.0, 1.0],
            [100.0, 1.0], [100.0, 0.0]
          ]
        ]
      },
      "properties": {
        "prop0": "value0",
        "prop1": { "this": "that" }
      }
    }

    geopolygon = bdm.GeoJSONObject(**polygon)

    assert geopolygon.type == 'Feature'
    assert geopolygon.geometry.type == 'Polygon'
    assert len(geopolygon.geometry.coordinates) == 1
    assert len(geopolygon.geometry.coordinates[0]) == 5
    assert len(geopolygon.geometry.coordinates[0][0]) == 2
    assert geopolygon.properties['prop0'] == 'value0'
    assert geopolygon.properties['prop1']['this'] == 'that'
    
    import logging
    logging.info(geopolygon.dict())