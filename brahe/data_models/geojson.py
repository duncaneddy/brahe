"""The geojson module provides data model classes for initialization and storing
of GeoJSON objects.
"""

import typing
import typing_extensions
import pydantic
import numpy as np

import brahe.astro as astro
import brahe.coordinates as coords
import brahe.frames as frames

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

    @property
    def geotype(self):
        '''Return GeoJSON geometry type.

        Returns:
            str: GeoJSON geometry type
        '''

        return self.geometry.type

    @property
    def num_points(self):
        '''Returns the number of unique points in the GeoJSON Object

        Returns:
            int: Number of points in object
        '''

        if self.geometry.type == 'Point':
            return 1
        elif self.geometry.type == 'LineString':
            return len(self.geometry.coordinates)
        elif self.geometry.type == 'Polygon':
            return len(self.geometry.coordinates[0]) - 1
        else:
            raise NotImplementedError(f'Function not implemented for GeoJSON Geometry type: {self.geotype}')

    @property
    def center(self):
        '''Return center point of object. Given at [lon, lat] in degrees. 

        Returns:
            np.ndarray: Center point of Geometry object. Units: [deg]
        '''
        if self.geometry.type == 'Point':
            return np.array(self.geometry.coordinates)
        elif self.geometry.type == 'LineString':
            center = np.zeros(len(self.geometry.coordinates[0]))

            for idx in range(0, self.num_points):
                center += self.geometry.coordinates[0][idx]/self.num_points

            return center
        elif self.geometry.type == 'Polygon':
            center = np.zeros(len(self.geometry.coordinates[0][0]))

            for idx in range(0, self.num_points):
                center += np.array(self.geometry.coordinates[0][idx])/self.num_points

            return center
        else:
            raise NotImplementedError(f'Function not implemented for GeoJSON Geometry type: {self.geotype}')

    @property
    def center_ecef(self):
        '''Return center point of object. Given as ecef coordinates.

        Returns:
            np.ndarray: Center point of Geometry object in ECEF frame. Units: [m]
        '''

        center = self.center

        # Ensure input has altitude
        if len(center) == 2:
            center = np.array([center[0], center[1], 0.0])

        ecef = coords.sGEODtoECEF(center, use_degrees=True)

        # Convert point to ECEF frame
        return ecef