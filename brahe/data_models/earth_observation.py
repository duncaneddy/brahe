"""The operations module provides data classes used in satellite operations and
task planning.
"""

import uuid
import enum
import datetime
import typing
import pydantic
import numpy as np

from brahe.epoch import Epoch
import brahe.coordinates as coords

from .geojson import GeoJSONObject

# Custom Type for UUID 4 validation
class StrUUID4(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        try:
            if type(v) == str:
                v = str(uuid.UUID(v, version=4))
            else:
                raise ValueError(f'"{v}" is not a valid UUID4 formatted string')
        except ValueError:
            raise ValueError(f'"{v}" is not a valid UUID4 formatted string')
        return v

##########################
# Earth Observation Base #
##########################

class EOBase(pydantic.BaseModel):
    '''Earth Observation base class.

    Provides the pydantic `Config` class type to set serializa
    '''
    class Config:
        json_encoders = {
            datetime.datetime: lambda v: v.strftime('%Y-%m-%dT%H:%M:%S.%f')
            [:-3] + 'Z',  # Truncated Milliseconds
            np.ndarray: lambda v: v.tolist()
        }

class TimeWindowProperties(EOBase):
    '''Add time window properties to an object. These are start-time, end-time,
    and mid-time.
    '''
    t_start: datetime.datetime = pydantic.Field(None, description='Start of time window.')
    t_mid: datetime.datetime = pydantic.Field(None, description='Time window mid-time.')
    t_end: datetime.datetime = pydantic.Field(None, description='End of time window.')
    t_duration: float = pydantic.Field(None, description='Time window duration.')

    # Need to strip timezone into
    @pydantic.validator('t_start', 't_mid', 't_end', 't_duration', always=True, pre=True)
    def handle_tzinfo(cls, value):
        return value.replace(tzinfo=None) if value and type(value) == datetime.datetime else value
    
    @pydantic.root_validator(pre=True)
    def set_times(cls, values):
        # Get possible inputs to set window object
        t_start = values.get('t_start')
        t_end = values.get('t_end')
        t_duration = values.get('t_duration')
        
        # Set start and end
        if t_start and t_end:
            # Start
            if type(t_start) == Epoch:
                t_start = t_start.to_datetime(tsys='UTC')
            elif type(t_start) == str:
                t_start = Epoch(t_start, tsys='UTC').to_datetime(tsys='UTC')

            # End
            if type(t_end) == Epoch:
                t_end = t_end.to_datetime(tsys='UTC')
            elif type(t_end) == str:
                t_end = Epoch(t_end, tsys='UTC').to_datetime(tsys='UTC')

            # Derive duration
            t_duration = (t_end - t_start).total_seconds()

            if t_end <= t_start:
                raise ValueError(f't_end must be after t_start')
        elif t_start and t_duration:
            if type(t_start) == Epoch:
                t_start = t_start.to_datetime(tsys='UTC')
            elif type(t_start) == str:
                t_start = Epoch(t_start, tsys='UTC').to_datetime(tsys='UTC')

            # Derive end time
            t_end = t_start + datetime.timedelta(seconds=t_duration)

        elif t_end and t_duration:
            if type(t_end) == Epoch:
                t_end = t_end.to_datetime(tsys='UTC')
            elif type(t_end) == str:
                t_end = Epoch(t_end, tsys='UTC').to_datetime(tsys='UTC')

            # Derive start time
            t_start = t_end - datetime.timedelta(seconds=t_duration)
        else:
            raise ValueError(f'Two of t_start, t_end, and t_duration must be provided.')

        # Set midtime
        t_mid = t_start + datetime.timedelta(seconds=t_duration) / 2.0

        # Populate in values
        values['t_start'] = t_start
        values['t_end'] = t_end
        values['t_duration'] = t_duration
        values['t_mid'] = t_mid

        # Return validation
        return values
        
    # Epoch Accessors
    @property
    def t_start_epc(self):
        '''Start time as `Epoch` type
        '''
        return Epoch(self.t_start, tsys='UTC')

    @property
    def t_end_epc(self):
        '''End time as `Epoch` type
        '''

        return Epoch(self.t_end, tsys='UTC')

    @property
    def t_mid_epc(self):
        '''Mid time as `Epoch` type
        '''

        return Epoch(self.t_mid, tsys='UTC')

#########
# Types #
#########

class ScheduleStatus(str, enum.Enum):
    '''Type to specify staus of item in schedule
    '''
    not_scheduled = 'not_scheduled'
    scheduled = 'scheduled'


class AscendingDescending(str, enum.Enum):
    '''Type to specify whether location access is ascending, descending, or either.
    '''
    ascending = 'ascending'
    descending = 'descending'
    either = 'either'

class LookDirection(str, enum.Enum):
    '''Whether collect is taken from a left-looking or right-looking direction.
    '''
    left = 'left'
    right = 'right'
    either = 'either'

######################
# Access Constraints #
######################

class AccessConstraints(EOBase):
    '''Class to store geometric access constraints for location observation.
    '''
    
    ascdsc: AscendingDescending = pydantic.Field(AscendingDescending.either, description='Constraint on access being taken during an ascending or descending pass.')
    look_direction: LookDirection = pydantic.Field(LookDirection.either, description='Constraint on access look direction.')
    look_angle_min: pydantic.confloat(ge=0.0,le=90.0) = pydantic.Field(0.0, description='Minimum look angle constraint.')
    look_angle_max: pydantic.confloat(ge=0.0,le=90.0) = pydantic.Field(90.0, description='Maximum look angle constraint.')
    elevation_min: pydantic.confloat(ge=0.0,le=90.0) = pydantic.Field(0.0, description='Minimum elevation constraint.')
    elevation_max: pydantic.confloat(ge=0.0,le=90.0) = pydantic.Field(90.0, description='Maximum elevation constraint.')

    @pydantic.validator('look_angle_max')
    def validate_look_angle(cls, look_angle_max, values):
        look_angle_min = values.get('look_angle_min')

        if look_angle_min != None and look_angle_min > look_angle_max:
            raise ValueError(f'Minimum look angle constraint ({look_angle_min}) must be less than the maximum constraint ({look_angle_max}).')

        return look_angle_max

    @pydantic.validator('elevation_max')
    def validate_elevation(cls, elevation_max, values):
        elevation_min = values.get('elevation_min')

        if elevation_min != None and elevation_min > elevation_max:
            raise ValueError(f'Minimum elevation constraint ({elevation_min}) must be less than the maximum constraint ({elevation_max}).')

        return elevation_max

class TessellationSettings(EOBase):
    '''Settings used to tesselation.
    '''
    tile_width: pydantic.confloat(ge=100.0) = pydantic.Field(20000.0, description='Maximum tile width')
    tile_length: pydantic.confloat(ge=100.0) = pydantic.Field(20000.0, description='Maximum tile length')
    cross_track_overlap: pydantic.confloat(ge=100.0) = pydantic.Field(100.0, description='Cross-track overlap.')
    along_track_overlap: pydantic.confloat(ge=100.0) = pydantic.Field(100.0, description='Along-track overlap.')

class AccessProperties(EOBase):
    ascdsc: typing.Optional[AscendingDescending] = None
    look_direction: typing.Optional[LookDirection] = None
    local_time: pydantic.confloat(ge=0, lt=86400) = None
    azimuth_open: float = None
    azimuth_close: float = None
    look_angle_min: float = None
    look_angle_max: float = None
    elevation_min: float = None
    elevation_max: float = None
    los_start: list = []
    los_end: list = []

    @pydantic.validator('los_start', 'los_end', pre=True)
    def validate_los(cls, los):
        if type(los) == np.ndarray:
            return los.tolist()
        else:
            return los


###########
# Request #
###########

class RequestProperties(EOBase):
    reward: pydantic.confloat(ge=0.0) = pydantic.Field(1.0, description='Request collection reward')
    id: StrUUID4 = pydantic.Field(None, description='Unique identifer for request')
    description: typing.Optional[str] = pydantic.Field('', description='Description of imaging request')
    constraints: AccessConstraints = pydantic.Field(AccessConstraints(), description='Access constraint requirements on accessing this location.')
    tessellation: TessellationSettings = pydantic.Field(TessellationSettings(), description='Tessellation settings.')
    collect_duration: pydantic.confloat(gt=0.0) = pydantic.Field(30.0, description='Duration of collection.')


    @pydantic.validator('id', pre=True, always=True)
    def set_id(cls, id, values):
        if not id:
            return str(uuid.uuid4())
        else:
            return id 

class Request(GeoJSONObject):
    properties: RequestProperties

    @property
    def id(self):
        '''Return Request ID

        Returns:
            str: Request identifier.
        '''
        return self.properties.id

    @property
    def request_id(self):
        '''Return Request ID

        Returns:
            str: Request identifier.
        '''
        return self.properties.id

    @property
    def description(self):
        '''Return Request description.

        Returns:
            str: Request description.
        '''
        return self.properties.description

    @property
    def constraints(self):
        '''Direct access of request constraints

        Returns:
            AccessConstraints: Request access constraints
        '''
        return self.properties.constraints

    @property
    def tessellation(self):
        '''Direct access of tessellation settings.

        Returns:
            TessellationSettings: Tessellation settings.
        '''
        return self.properties.tessellation

    @property
    def reward(self):
        '''Request collection reward

        Returns:
            Reward for collection
        '''
        return self.properties.reward

    

# ########
# # Tile #
# ########

class TileProperties(EOBase):
    id: StrUUID4 = pydantic.Field(None, description='Unique identifer for tile')
    tile_group_id: StrUUID4 = pydantic.Field(None, description='Unique identifer for tile group of the tile')
    request_id: StrUUID4 = pydantic.Field(..., description='Unique identifer for request')
    spacecraft_ids: typing.List[pydantic.conint(ge=1)] = pydantic.Field([], description='Spacecraft IDs that can collect this tile.')
    tile_direction: pydantic.conlist(float, min_items=3, max_items=3) = pydantic.Field(..., description='Direction of tiling. Normalized unit vector in Cartesian ECEF frame.')
    
    @pydantic.validator('id', pre=True, always=True)
    def set_id(cls, id, values):
        if not id:
            return str(uuid.uuid4())
        else:
            return id

class Tile(GeoJSONObject):
    properties: TileProperties

    @property
    def id(self):
        '''Return Tile ID

        Returns:
            UUID4: Tile identifier.
        '''
        return self.properties.id

    @property
    def tile_id(self):
        '''Return Tile ID

        Returns:
            UUID4: Tile identifier.
        '''
        return self.properties.id

    @property
    def tile_group_id(self):
        '''Return Tile Group ID

        Returns:
            UUID4: Tile Group identifier.
        '''
        return self.properties.tile_group_id

    @property
    def request_id(self):
        '''Return Request ID

        Returns:
            UUID4: Request identifier.
        '''
        return self.properties.request_id

    @property
    def spacecraft_ids(self):
        '''Return Satellite IDs

        Returns:
            UUID4: Statellite identifiers.
        '''
        return self.properties.spacecraft_ids

    @property
    def tile_direction(self):
        '''Quick access for Tile Direction

        Returns:
            List: Tile direction
        '''
        return self.properties.tile_direction

# ###########
# # Station #
# ###########

class StationProperties(EOBase):
    id: StrUUID4 = pydantic.Field(None, description='Unique identifer for station')
    station_name: str = pydantic.Field('', description='Name of station')
    constraints: AccessConstraints = pydantic.Field(AccessConstraints(), description='Constraints on accessing station')
    downlink_rate_max: pydantic.confloat() = pydantic.Field(0.0, description='Maximum downlink datarate at station. Units: [GB/s]')
    

    @pydantic.validator('id', pre=True, always=True)
    def set_id(cls, id, values):
        if not id:
            return str(uuid.uuid4())
        else:
            return id

class Station(GeoJSONObject):
    properties: StationProperties

    @property
    def id(self):
        '''Return Station ID

        Returns:
            UUID4: Station identifier.
        '''
        return self.properties.id

    @property
    def station_id(self):
        '''Return Station ID

        Returns:
            UUID4: Station identifier.
        '''
        return self.properties.id

    @property
    def station_name(self):
        '''Direct access to station name.

        Returns:
            str: Station name.
        '''
        return self.properties.station_name

    @property
    def constraints(self):
        '''Direct access of station constraints

        Returns:
            AccessConstraints: Station access constraints
        '''
        return self.properties.constraints

    @property
    def downlink_rate_max(self):
        '''Return Station downlink datarate

        Returns:
            float: Station downlink datarate
        '''
        return self.properties.downlink_rate_max

###############
# Opportunity #
###############

class Opportunity(TimeWindowProperties):
    id: StrUUID4 = pydantic.Field(None, description='Unique identifer of opportunity')
    spacecraft_id: typing.Union[pydantic.conint(ge=1)] = pydantic.Field(..., description='ID of spacecraft associated with opportunity.')
    status: ScheduleStatus = pydantic.Field('not_scheduled', description='Status of opportunity in schedule.')
    center: pydantic.conlist(float, min_items=2, max_items=3) = pydantic.Field(..., description='Center Geodetic Point of Opportunity')
    center_ecef: pydantic.conlist(float, min_items=3, max_items=3) = pydantic.Field(None, description='Center ECEF Point of Opportunity')
    access_properties: AccessProperties = pydantic.Field(AccessProperties(), descripton='Properties associated with collection')

    @pydantic.validator('id', pre=True, always=True)
    def set_id(cls, id, values):
        if not id:
            return str(uuid.uuid4())
        else:
            return id

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
# Collect #
###########

class Collect(Opportunity):
    request_id: StrUUID4 = pydantic.Field(..., description='Unique identifer for request')
    tile_id: StrUUID4 = pydantic.Field(None, description='Unique identifer for tile')
    tile_group_id: StrUUID4 = pydantic.Field(None, description='Unique identifer for tile group')
    reward: pydantic.confloat(ge=0.0) = pydantic.Field(1.0, description='Request collection reward')

    @property
    def collect_id(self):
        '''Access alternate name for collect ID
        '''
        return self.id

###########
# Contact #
###########

class Contact(Opportunity):
    station_id: StrUUID4 = pydantic.Field(..., description='Unique identifer for station')
    station_name: str = pydantic.Field('', description='Name of Station')

    @property
    def contact_id(self):
        '''Access alternate name for contact ID
        '''
        return self.id