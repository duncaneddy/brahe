# -*- coding: utf-8 -*-
"""The geojson module provides data model classes for initialization and storing
of GeoJSON objects.
"""

import typing
import typing_extensions
import pydantic

geographic_point = pydantic.conlist(pydantic.confloat(ge=-180,le=180), min_items=2, max_items=2)

class GeoJSONGeometry(pydantic.BaseModel):
    type: typing_extensions.Literal['Point', 'LineString', 'Polygon', 'MultiPoint', 'MultiLineString', 'MultiPolygon']
    coordinates: typing.Union[geographic_point, 
        typing.List[geographic_point], 
        typing.List[typing.List[geographic_point]], 
        typing.List[typing.List[typing.List[geographic_point]]]] = pydantic.Field(..., description='Geomtry Coordinates')

class GeoJSONObject(pydantic.BaseModel):
    type: typing_extensions.Literal['Feature'] = pydantic.Field('Feature', description='GeoJSON Object type')
    geometry: GeoJSONGeometry = pydantic.Field('Feature', description='GeoJSON object type')
    properties: typing.Optional[dict] = pydantic.Field({}, description='Additional properties')