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
curl -L https://datacenter.iers.org/data/latestVersion/finals.all.iau2000.txt -o ./data/finals.all.iau2000.txt
curl -L https://datacenter.iers.org/data/latestVersion/EOP_20_C04_one_file_1962-now.txt -o ./data/EOP_20_C04_one_file_1962-now.txt

# Space Weather Data
curl -L https://celestrak.com/SpaceData/sw19571001.txt -o ./data/sw19571001.txt
curl -L https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt -o ./data/fluxtable.txt
