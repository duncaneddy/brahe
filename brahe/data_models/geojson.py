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
from pydantic import Field
from typing import List
from typing_extensions import Annotated

geographic_point = Annotated[List[float], Field(min_length=2, max_length=3)]

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

    @property
    def coordinates(self):
        '''Returns the coordinates of the GeoJSON Object. If polygon type
        the last point will be repeated.
        '''

        if self.geometry.type == 'Point':
            pnt = self.geometry.coordinates
            if len(pnt) == 2:
                yield np.array([pnt[0], pnt[1], 0.0])
            else:
                yield np.array(pnt)
        elif self.geometry.type == 'Polygon':
            if len(self.geometry.coordinates) > 1:
                raise RuntimeError('Polygon with multiple lines are not currently supported.')

            for idx in range(0, self.num_points + 1):
                pnt = self.geometry.coordinates[0][idx]
                if len(pnt) == 2:
                    yield np.array([pnt[0], pnt[1], 0.0])
                else:
                    yield np.array(pnt)
        else:
            raise NotImplementedError(f'Function not implemented for GeoJSON Geometry type: {self.geotype}')

    @property
    def vertices(self):
        '''Returns the unique vertices of the GeoJSON Object. This ensures
        for polygon types the last point won't be repeated.
        '''

        if self.geometry.type == 'Point':
            pnt = self.geometry.coordinates
            if len(pnt) == 2:
                yield np.array([pnt[0], pnt[1], 0.0])
            else:
                yield np.array(pnt)

        elif self.geometry.type == 'Polygon':
            if len(self.geometry.coordinates) > 1:
                raise RuntimeError('Polygon with multiple lines are not currently supported.')

            for idx in range(0, self.num_points):
                pnt = self.geometry.coordinates[0][idx]
                if len(pnt) == 2:
                    yield np.array([pnt[0], pnt[1], 0.0])
                else:
                    yield np.array(pnt)
        else:
            raise NotImplementedError(f'Function not implemented for GeoJSON Geometry type: {self.geotype}')