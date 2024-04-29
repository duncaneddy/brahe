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
from brahe.tle import TLE
from pydantic import Field
from typing_extensions import Annotated

class SpacecraftModel(pydantic.BaseModel):
    '''Spacecraft Dynamics and resource model
    '''

    slew_rate: float = pydantic.Field(1.0, description='Maximum spacecraft slew rate. Units: [deg/s]')
    settling_time: float = pydantic.Field(15.0, description='Pointing stellting time after slew.')
    power_max: float = pydantic.Field(0.0, description='Maximum on-board power storage. Untis: [Joules]')
    power_min: float = pydantic.Field(0.0, description='Minimum recoverable spacecraft power. Units: [Joules]')
    power_rate_payload: float = pydantic.Field(0.0,description='Power rate when using payload. Units: [Watts]')
    power_rate_downlink: float = pydantic.Field(0.0,description='Power rate when downliking. Units: [Watts]')
    power_rate_idle: float = pydantic.Field(0.0,description='Power rate when idle. Units: [Watts]')
    power_rate_sunpoint: float = pydantic.Field(0.0,description='Power rate when Sun-pointed. Units: [Watts]')
    data_max: float = pydantic.Field(0.0,description='Total maximum data storage of spacecraft. Units: [GigaBytes]')
    data_rate_payload: float = pydantic.Field( 0.0, description='Spacecraft data rates when using payload. All units in [GB/s]')
    data_rate_downlink: float = pydantic.Field( 0.0, description='Spacecraft data rates when downlinking. All units in [GB/s]')

class Spacecraft(pydantic.BaseModel):
    id: Annotated[int, Field(ge=0)] = pydantic.Field(None, description='Spacecraft Identifider')
    name: str = pydantic.Field('', description='Spacecraft name')
    line1: typing.Optional[str] = pydantic.Field(None, description='First line of TLE associated with spacecraft [Optional]')
    line2: typing.Optional[str] = pydantic.Field(None, description='Second line of TLE associated with spacecraft [Optional]')
    model: typing.Optional[SpacecraftModel] = pydantic.Field(None, description='Resource model for spacecraft')

    @property
    def spacecraft_id(self):
        '''Alternate name for Spacecraft ID
        '''
        return self.id

    @property
    def tle(self):
        '''TLE object for spacecraft
        '''
        return TLE(self.line1, self.line2)
