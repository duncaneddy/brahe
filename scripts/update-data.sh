#!/bin/bash
# 
# Description:
#
#   Update Earth orientation and space weather data
#
# Usage: 
#   $ ./update-data.sh
# 

# IERS Earth Orientation Data
# Sourced from USNO and Paris Observatory (IERS) mirrors; the primary IERS
# datacenter (datacenter.iers.org) is frequently unavailable.
curl -fL https://maia.usno.navy.mil/ser7/finals2000A.all -o ./data/eop/finals.all.iau2000.txt
curl -fL https://hpiers.obspm.fr/iers/eop/eopc04/eopc04.1962-now -o ./data/eop/EOP_C04_one_file_1962-now.txt

# Space Weather Data
curl -L https://celestrak.org/SpaceData/sw19571001.txt -o ./data/space_weather/sw19571001.txt
curl -L https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt -o ./data/space_weather/fluxtable.txt
