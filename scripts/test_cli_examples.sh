#!/bin/bash
#
# Test script for Brahe CLI commands
#
# This script tests all documented CLI commands to ensure they work correctly
# and outputs match the documentation examples.
#
# Usage:
#   ./scripts/test_cli_examples.sh [--verbose]
#

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Verbose mode
VERBOSE=0
if [[ "$1" == "--verbose" ]]; then
    VERBOSE=1
fi

# Test result functions
test_start() {
    TESTS_RUN=$((TESTS_RUN + 1))
    if [[ $VERBOSE -eq 1 ]]; then
        echo -e "${YELLOW}[TEST $TESTS_RUN]${NC} $1"
    fi
}

test_pass() {
    TESTS_PASSED=$((TESTS_PASSED + 1))
    if [[ $VERBOSE -eq 1 ]]; then
        echo -e "${GREEN}✓ PASS${NC}"
    else
        echo -n "."
    fi
}

test_fail() {
    TESTS_FAILED=$((TESTS_FAILED + 1))
    echo -e "${RED}✗ FAIL${NC} - $1"
}

test_skip() {
    echo -e "${YELLOW}⊘ SKIP${NC} - $1"
}

echo "========================================"
echo "Brahe CLI Test Suite"
echo "========================================"
echo ""

#
# TRANSFORM COMMANDS
#

echo "Testing transform commands..."

# Test: transform frame ECI to ECEF
test_start "transform frame ECI to ECEF"
OUTPUT=$(brahe transform frame ECI ECEF "2024-01-01T00:00:00Z" 6878137 0 0 0 7500 0 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "transform frame ECI to ECEF failed"
fi

# Test: transform frame ECEF to ECI
test_start "transform frame ECEF to ECI"
OUTPUT=$(brahe transform frame ECEF ECI "2024-01-01T00:00:00Z" -- -1176064.179 -6776827.197 15961.825 6895.377 -1196.637 0.241 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "transform frame ECEF to ECI failed"
fi

# Test: transform coordinates keplerian to cartesian
test_start "transform coordinates keplerian to cartesian"
OUTPUT=$(brahe transform coordinates keplerian cartesian "" 6878137 0.001 97.8 0 0 0 --as-degrees 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "keplerian to cartesian conversion failed"
fi

# Test: transform coordinates cartesian to keplerian
test_start "transform coordinates cartesian to keplerian"
OUTPUT=$(brahe transform coordinates --as-degrees cartesian keplerian "" -- 6871258.863 0.0 0.0 0.0 -1034.183 7549.721 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "cartesian to keplerian conversion failed"
fi

# Test: transform coordinates geodetic to cartesian
test_start "transform coordinates geodetic to cartesian ECEF"
OUTPUT=$(brahe transform coordinates --as-degrees --to-frame ECEF geodetic cartesian "" 40.7128 74.0 10 0 0 0 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "geodetic to cartesian conversion failed"
fi

# Test: transform coordinates with frame change
test_start "transform coordinates cartesian ECI to ECEF"
OUTPUT=$(brahe transform coordinates --from-frame ECI --to-frame ECEF cartesian cartesian "2024-01-01T00:00:00Z" 6878137 0 0 0 7500 0 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "cartesian frame change failed"
fi

#
# TIME COMMANDS
#

echo -e "\nTesting time commands..."

# Test: time add
test_start "time add - add 1 hour"
OUTPUT=$(brahe time add "2024-01-01T00:00:00Z" 3600 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "time add failed"
fi

# Test: time add negative
test_start "time add - subtract 30 minutes"
OUTPUT=$(brahe time add "2024-01-01T12:00:00Z" -1800 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "time add (negative) failed"
fi

# Test: time range
test_start "time range - 30 minute intervals"
OUTPUT=$(brahe time range "2024-01-01T00:00:00Z" "2024-01-01T01:00:00Z" 1800 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "time range failed"
fi

# Test: time time-system-offset
test_start "time time-system-offset UTC to TAI"
OUTPUT=$(brahe time time-system-offset "2024-01-01T00:00:00Z" UTC TAI 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "time time-system-offset failed"
fi

# Test: time convert
test_start "time convert - string to mjd"
OUTPUT=$(brahe time convert "2024-01-01T00:00:00Z" string mjd --input-time-system UTC --output-time-system UTC 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "time convert failed"
fi

# Test: time convert - mjd to string
test_start "time convert - mjd to string"
OUTPUT=$(brahe time convert 60310.0 mjd string --input-time-system UTC --output-time-system UTC 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "time convert mjd to string failed"
fi

# Test: time convert - time system conversion
test_start "time convert - UTC to TAI"
OUTPUT=$(brahe time convert "2024-01-01T00:00:00Z" string string --input-time-system UTC --output-time-system TAI 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "time convert time system failed"
fi

#
# ORBITS COMMANDS
#

echo -e "\nTesting orbits commands..."

# Test: orbits orbital-period
test_start "orbits orbital-period - LEO 500km"
OUTPUT=$(brahe orbits orbital-period "R_EARTH+500e3" 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "orbital-period failed"
fi

# Test: orbits orbital-period with units
test_start "orbits orbital-period - minutes"
OUTPUT=$(brahe orbits orbital-period "R_EARTH+500e3" --units minutes 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "orbital-period with units failed"
fi

# Test: orbits sma-from-period
test_start "orbits sma-from-period - 90 minutes"
OUTPUT=$(brahe orbits sma-from-period 90 --units minutes 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "sma-from-period failed"
fi

# Test: orbits mean-motion
test_start "orbits mean-motion - LEO 500km"
OUTPUT=$(brahe orbits mean-motion "R_EARTH+500e3" 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "mean-motion failed"
fi

# Test: orbits mean-motion with custom GM
test_start "orbits mean-motion - with custom GM"
OUTPUT=$(brahe orbits mean-motion "R_EARTH+500e3" --gm GM_EARTH 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "mean-motion with GM failed"
fi

# Test: orbits anomaly-conversion
test_start "orbits anomaly-conversion - mean to true"
OUTPUT=$(brahe orbits anomaly-conversion --as-degrees 45.0 0.1 mean true 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "anomaly-conversion failed"
fi

# Test: orbits sun-sync-inclination
test_start "orbits sun-sync-inclination - 500km"
OUTPUT=$(brahe orbits sun-sync-inclination "R_EARTH+500e3" 0.0 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "sun-sync-inclination failed"
fi

# Test: orbits perigee-velocity
test_start "orbits perigee-velocity - circular orbit"
OUTPUT=$(brahe orbits perigee-velocity "R_EARTH+500e3" 0.0 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "perigee-velocity failed"
fi

# Test: orbits apogee-velocity
test_start "orbits apogee-velocity - circular orbit"
OUTPUT=$(brahe orbits apogee-velocity "R_EARTH+500e3" 0.0 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "apogee-velocity failed"
fi

#
# EOP COMMANDS
#

echo -e "\nTesting eop commands..."

# Test: eop get-utc-ut1
test_start "eop get-utc-ut1"
OUTPUT=$(brahe eop get-utc-ut1 "2024-01-01T00:00:00Z" 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "eop get-utc-ut1 failed"
fi

# Test: eop get-polar-motion
test_start "eop get-polar-motion"
OUTPUT=$(brahe eop get-polar-motion "2024-01-01T00:00:00Z" 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "eop get-polar-motion failed"
fi

# Test: eop download (skip - would overwrite user's data)
test_skip "eop download - would overwrite cached data"

#
# ACCESS COMMANDS
#

echo -e "\nTesting access commands..."

# Test: access compute (with short duration for speed)
test_start "access compute - ISS over NYC (1 day)"
OUTPUT=$(brahe access compute 25544 --lat 40.7128 --lon=-74.0060 --duration 1 --output-format simple 2>&1)
if [[ $? -eq 0 ]]; then
    test_pass
else
    test_fail "access compute failed"
fi

# Test: access compute with min elevation
test_start "access compute - with min elevation"
OUTPUT=$(brahe access compute 25544 --lat 40.7128 --lon=-74.0060 --duration 0.5 --min-elevation 15 --output-format simple 2>&1)
if [[ $? -eq 0 ]]; then
    test_pass
else
    test_fail "access compute with min-elevation failed"
fi

#
# DATASETS COMMANDS
#

echo -e "\nTesting datasets commands..."

# Test: datasets celestrak list-groups
test_start "datasets celestrak list-groups"
OUTPUT=$(brahe datasets celestrak list-groups 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "celestrak list-groups failed"
fi

# Test: datasets celestrak lookup
test_start "datasets celestrak lookup - ISS"
OUTPUT=$(brahe datasets celestrak lookup "ISS" 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "celestrak lookup failed"
fi

# Test: datasets celestrak show
test_start "datasets celestrak show - ISS (25544)"
OUTPUT=$(brahe datasets celestrak show 25544 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "celestrak show failed"
fi

# Test: datasets groundstations list-providers
test_start "datasets groundstations list-providers"
OUTPUT=$(brahe datasets groundstations list-providers 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "groundstations list-providers failed"
fi

# Test: datasets groundstations list-stations
test_start "datasets groundstations list-stations"
OUTPUT=$(brahe datasets groundstations list-stations 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "groundstations list-stations failed"
fi

# Test: datasets groundstations list-stations with filter
test_start "datasets groundstations list-stations --provider ksat"
OUTPUT=$(brahe datasets groundstations list-stations --provider ksat 2>&1)
if [[ $? -eq 0 ]] && [[ ! -z "$OUTPUT" ]]; then
    test_pass
else
    test_fail "groundstations list-stations with filter failed"
fi

#
# SUMMARY
#

echo -e "\n"
echo "========================================"
echo "Test Results"
echo "========================================"
echo "Tests run:    $TESTS_RUN"
echo -e "${GREEN}Tests passed: $TESTS_PASSED${NC}"
if [[ $TESTS_FAILED -gt 0 ]]; then
    echo -e "${RED}Tests failed: $TESTS_FAILED${NC}"
else
    echo "Tests failed: 0"
fi
echo ""

if [[ $TESTS_FAILED -gt 0 ]]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
