"""The operations module provides data classes used in satellite operations and
task planning.
"""

import uuid
import enum
import datetime
import typing
import pydantic
import numpy as np

import brahe.astro as astro
import brahe.coordinates as coords
import brahe.frames as frames

from .geojson import GeoJSONObject

#########
# Types #
#########

class AscendingDescending(enum.Enum):
    '''Type to specify whether location access is ascending, descending, or either.
    '''
    ascending = 'ascending'
    descending = 'descending'
    either = 'either'

###########
# Request #
###########

class RequestProperties(pydantic.BaseModel):
    ascdsc: AscendingDescending = pydantic.Field(AscendingDescending.either, description='Specify whether')
    look_angle_min: pydantic.confloat(ge=0.0,le=90.0) = pydantic.Field(0.0, description='Minimum look angle')
    look_angle_max: pydantic.confloat(ge=0.0,le=90.0) = pydantic.Field(50, description='Maximum look angle')
    reward: pydantic.confloat(ge=0.0) = pydantic.Field(1.0, description='Tasking request collection reward')
    request_id: typing.Union[pydantic.conint(ge=1), pydantic.UUID4] = pydantic.Field(None, description='Unique identifer for tasking request')

    class Config:
        # Define custom datetime serialization encoding
        json_encoders = {
            datetime.datetime: lambda dt: dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        }

    @pydantic.validator('request_id', pre=True, always=True)
    def set_id(cls, request_id, values):
        if not request_id:
            return str(uuid.uuid4())
        else:
            return request_id

class Request(GeoJSONObject):
    properties: RequestProperties

    @property
    def id(self):
        '''Return Request ID

        Returns:
            Union[uuid.uuid4, int]: Request identifier.
        '''
        return self.properties.request_id

    @property
    def request_id(self):
        '''Return Request ID

        Returns:
            Union[uuid.uuid4, int]: Request identifier.
        '''
        return self.properties.request_id

    @property
    def reward(self):
        '''Return Request reward

        Returns:
            float: Request reward
        '''
        return self.properties.reward

    @property
    def look_angle_min(self):
        '''Return Request minimum look angle

        Returns:
            float: Request minimum look angle
        '''
        return self.properties.look_angle_min

    @property
    def look_angle_max(self):
        '''Return Request maximum look angle

        Returns:
            float: Request maximum look angle
        '''
        return self.properties.look_angle_max

    @property
    def ascdsc(self):
        '''Return Request ascending/descending constraint

        Returns:
            float: Request ascending/descending constraint
        '''
        return self.properties.ascdsc

########
# Tile #
########

class TileProperties(pydantic.BaseModel):
    ascdsc: AscendingDescending = pydantic.Field(..., description='Specify whether')
    request_id: typing.Union[pydantic.conint(ge=1), pydantic.UUID4] = pydantic.Field(..., description='Unique identifer for request')
    tile_id: typing.Union[pydantic.conint(ge=1), pydantic.UUID4] = pydantic.Field(None, description='Unique identifer for tile')
    sat_ids: typing.List[typing.Union[pydantic.conint(ge=1), pydantic.UUID4]] = pydantic.Field(..., description='Unique identifer for satellites that can collect this tile.')

    class Config:
        # Define custom datetime serialization encoding
        json_encoders = {
            datetime.datetime: lambda dt: dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        }

    @pydantic.validator('tile_id', pre=True, always=True)
    def set_id(cls, tile_id, values):
        if not tile_id:
            return str(uuid.uuid4())
        else:
            return tile_id

class Tile(GeoJSONObject):
    properties: TileProperties

    @property
    def request_id(self):
        '''Return Request ID

        Returns:
            Union[uuid.uuid4, int]: Request identifier.
        '''
        return self.properties.request_id

    @property
    def sat_ids(self):
        '''Return Satellite IDs

        Returns:
            Union[uuid.uuid4, int]: Statellite identifiers.
        '''
        return self.properties.sat_ids

    @property
    def tile_id(self):
        '''Return TIle ID

        Returns:
            Union[uuid.uuid4, int]: Sat identifier.
        '''
        return self.properties.tile_id

    @property
    def id(self):
        '''Return TIle ID

        Returns:
            Union[uuid.uuid4, int]: Sat identifier.
        '''
        return self.properties.tile_id

    @property
    def ascdsc(self):
        '''Return whether tile is associated with ascending or descending pass

        Returns:
            float: Station cost per minute
        '''
        return self.properties.ascdsc

###########
# Station #
###########

class StationProperties(pydantic.BaseModel):
    elevation_min: pydantic.confloat(ge=0.0,le=90.0) = pydantic.Field(5.0, description='Minimum elevation')
    cost_per_min: pydantic.confloat(le=0.0) = pydantic.Field(0.0, description='Cost per minute for usage.')
    downlink_datarate: pydantic.confloat() = pydantic.Field(0.0, description='Downlink datarate')
    station_id: typing.Union[pydantic.conint(ge=1), pydantic.UUID4] = pydantic.Field(None, description='Unique identifer for station')

    class Config:
        # Define custom datetime serialization encoding
        json_encoders = {
            datetime.datetime: lambda dt: dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        }

    @pydantic.validator('station_id', pre=True, always=True)
    def set_id(cls, station_id, values):
        if not station_id:
            return str(uuid.uuid4())
        else:
            return station_id

class Station(GeoJSONObject):
    properties: StationProperties

    @property
    def station_id(self):
        '''Return Station ID

        Returns:
            Union[uuid.uuid4, int]: Station identifier.
        '''
        return self.properties.station_id

    @property
    def id(self):
        '''Return Station ID

        Returns:
            Union[uuid.uuid4, int]: Station identifier.
        '''
        return self.properties.station_id

    @property
    def cost_per_min(self):
        '''Return Station cost per minute

        Returns:
            float: Station cost per minute
        '''
        return self.properties.cost_per_min

    @property
    def downlink_datarate(self):
        '''Return Station downlink datarate

        Returns:
            float: Station downlink datarate
        '''
        return self.properties.downlink_datarate

    @property
    def elevation_min(self):
        '''Return Station minimum elevation

        Returns:
            float: Station minimum elevation
        '''
        return self.properties.elevation_min

###############
# Opportunity #
###############

class Opportunity(pydantic.BaseModel):
    opportunity_id: typing.Union[pydantic.conint(ge=1), pydantic.UUID4] = pydantic.Field(None, description='Unique identifer for opportunity')
    t_start: datetime.datetime = pydantic.Field(..., description='Start of opportunity')
    t_end: datetime.datetime = pydantic.Field(..., description='End of opportunity')
    t_mid: datetime.datetime = pydantic.Field(None, description='Opportunity mid-time')
    t_duration: float = pydantic.Field(None, description='Time duration of opportunity')
    sat_id: typing.Union[pydantic.conint(ge=1), pydantic.UUID4] = pydantic.Field(..., description='Unique identifer for the satellite.')


    class Config:
        # Define custom datetime serialization encoding
        json_encoders = {
            datetime.datetime: lambda dt: dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        }

    @pydantic.validator('t_mid', pre=True, always=True)
    def set_t_mid(cls, t_mid, values):
        if not t_mid:
            return values['t_start'] + datetime.timedelta(seconds=(values['t_end'] - values['t_start']).total_seconds()/2.0)
        else:
            return t_mid

    @pydantic.validator('t_duration', pre=True, always=True)
    def set_t_duration(cls, t_duration, values):
        if not t_duration:
            return (values['t_end'] - values['t_start']).total_seconds()
        else:
            return t_duration

    @pydantic.validator('opportunity_id', pre=True, always=True)
    def set_id(cls, opportunity_id, values):
        if not opportunity_id:
            return str(uuid.uuid4())
        else:
            return opportunity_id

    @property
    def id(self):
        '''Return Opportunity ID

        Returns:
            Union[uuid.uuid4, int]: Station identifier.
        '''
        return self.opportunity_id

###########
# Collect #
###########

class Collect(Opportunity):
    center: pydantic.conlist(float, min_items=2, max_items=3) = pydantic.Field(..., description='Center Geodetic Point for associated tile')
    center_ecef: pydantic.conlist(float, min_items=3, max_items=3) = pydantic.Field(None, description='Center ECEF Point for associated tile')
    request_id: typing.Union[pydantic.conint(ge=1), pydantic.UUID4] = pydantic.Field(..., description='Unique identifer for request')
    tile_id: typing.Union[pydantic.conint(ge=1), pydantic.UUID4] = pydantic.Field(None, description='Unique identifer for tile')
    look_angle_min: pydantic.confloat(ge=0.0,le=90.0) = pydantic.Field(0.0, description='Minimum look angle during collect')
    look_angle_max: pydantic.confloat(ge=0.0,le=90.0) = pydantic.Field(50, description='Maximum look angle during collect')
    reward: pydantic.confloat(ge=0.0) = pydantic.Field(1.0, description='Tasking request collection reward')

    @pydantic.validator('center_ecef', pre=True, always=True)
    def set_center_ecef(cls, center_ecef, values):
        if not center_ecef:
            center = values['center']
            # Ensure input has altitude
            if len(center) == 2:
                center = np.array([center[0], center[1], 0.0])

            ecef = coords.sGEODtoECEF(center, use_degrees=True)

            # Convert point to ECEF frame
            return ecef.tolist()
        else:
            return center_ecef

###########
# Contact #
###########

class Contact(Opportunity):
    center: pydantic.conlist(float, min_items=2, max_items=3) = pydantic.Field(..., description='Center Geodetic Point for associated tile')
    center_ecef: pydantic.conlist(float, min_items=3, max_items=3) = pydantic.Field(None, description='Center ECEF Point for associated tile')
    station_id: typing.Union[pydantic.conint(ge=1), pydantic.UUID4] = pydantic.Field(..., description='Unique identifer for station')
    elevation_min: pydantic.confloat(ge=0.0,le=90.0) = pydantic.Field(0.0, description='Minimum elevation during collect')
    elevation_max: pydantic.confloat(ge=0.0,le=90.0) = pydantic.Field(50, description='Maximum elevation during collect')

    @pydantic.validator('center_ecef', pre=True, always=True)
    def set_center_ecef(cls, center_ecef, values):
        if not center_ecef:
            center = values['center']
            # Ensure input has altitude
            if len(center) == 2:
                center = np.array([center[0], center[1], 0.0])

            ecef = coords.sGEODtoECEF(center, use_degrees=True)

            # Convert point to ECEF frame
            return ecef.tolist()
        else:
            return center_ecef