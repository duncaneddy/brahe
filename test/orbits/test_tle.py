"""Tests for TLE (Two-Line Element) functionality."""

import pytest
import numpy as np
import brahe


class TestTLEChecksum:
    """Test TLE checksum calculation."""

    @pytest.mark.parametrize("line,expected", [
        ("1 20580U 90037B   25261.05672437  .00006481  00000+0  23415-3 0  9990", 0),
        ("1 24920U 97047A   25261.00856804  .00000165  00000+0  89800-4 0  9991", 1),
        ("1 00900U 64063C   25261.21093924  .00000602  00000+0  60787-3 0  9992", 2),
        ("1 26605U 00071A   25260.44643294  .00000025  00000+0  00000+0 0  9993", 3),
        ("2 26410 146.0803  17.8086 8595307 233.2516   0.1184  0.44763667 19104", 4),
        ("1 28414U 04035B   25261.30628127  .00003436  00000+0  25400-3 0  9995", 5),
        ("1 28371U 04025F   25260.92882365  .00000356  00000+0  90884-4 0  9996", 6),
        ("1 19751U 89001C   25260.63997541  .00000045  00000+0  00000+0 0  9997", 7),
        ("1 29228U 06021A   25261.14661065  .00002029  00000+0  12599-3 0  9998", 8),
        ("2 31127  98.3591 223.5782 0064856  30.4095 330.0844 14.63937036981529", 9),
    ])
    def test_calculate_tle_line_checksum(self, line, expected):
        """Test TLE line checksum calculation."""
        checksum = brahe.calculate_tle_line_checksum(line)
        assert checksum == expected


class TestTLELineValidation:
    """Test TLE line validation."""

    @pytest.mark.parametrize("line", [
        "1 20580U 90037B   25261.05672437  .00006481  00000+0  23415-3 0  9990",
        "1 24920U 97047A   25261.00856804  .00000165  00000+0  89800-4 0  9991",
        "1 00900U 64063C   25261.21093924  .00000602  00000+0  60787-3 0  9992",
        "1 26605U 00071A   25260.44643294  .00000025  00000+0  00000+0 0  9993",
        "2 26410 146.0803  17.8086 8595307 233.2516   0.1184  0.44763667 19104",
        "1 28414U 04035B   25261.30628127  .00003436  00000+0  25400-3 0  9995",
        "1 28371U 04025F   25260.92882365  .00000356  00000+0  90884-4 0  9996",
        "1 19751U 89001C   25260.63997541  .00000045  00000+0  00000+0 0  9997",
        "1 29228U 06021A   25261.14661065  .00002029  00000+0  12599-3 0  9998",
        "2 31127  98.3591 223.5782 0064856  30.4095 330.0844 14.63937036981529",
    ])
    def test_validate_tle_line_valid(self, line):
        """Test validation of valid TLE lines."""
        assert brahe.validate_tle_line(line)

    @pytest.mark.parametrize("line", [
        "1 20580U 90037B   25261.05672437  .00006481  00000+0  23415-3 0  9980",  # Wrong checksum
        "1 24920U 97047A   25261.00856804  .00000165  00000+0  89800-4 0  9931",  # Wrong checksum
        "1 00900U 64063C   25261.21093924  .00000602  00000+0  60787-3 0  9912",  # Wrong checksum
        "1 26605U 00071A   25260.44643294  .00000025  00000+0  00000+0 0  9983",  # Wrong checksum
        "2 26410 146.0803  17.8086 8595307 233.2516   0.1184  19104",           # Too short
        "1 28414U 04035B   25261.30628127  .00003436  00000+0  25400-3 0  9923421295",  # Too long
        "3 28371U 04025F   25260.92882365  .00000356  00000+0  90884-4 0  9996",  # Wrong line number
        "3 19751U 89001C   25260.63997541  .00000045  00000+0  00000+0 0  9999",  # Wrong line number
    ])
    def test_validate_tle_line_invalid(self, line):
        """Test validation of invalid TLE lines."""
        assert not brahe.validate_tle_line(line)


class TestTLELinesValidation:
    """Test TLE line pair validation."""

    @pytest.mark.parametrize("line1,line2", [
        (
            "1 22195U 92070B   25260.83452377 -.00000009  00000+0  00000+0 0  9999",
            "2 22195  52.6519  78.7552 0137761  68.4365 290.4819  6.47293897777784"
        ),
        (
            "1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9997",
            "2 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110516"
        ),
    ])
    def test_validate_tle_lines_valid(self, line1, line2):
        """Test validation of valid TLE line pairs."""
        assert brahe.validate_tle_lines(line1, line2)

    @pytest.mark.parametrize("line1,line2", [
        # Mismatched NORAD IDs
        (
            "1 22195U 92070B   25260.83452377 -.00000009  00000+0  00000+0 0  9999",
            "2 22196  52.6519  78.7552 0137761  68.4365 290.4819  6.47293897777784"
        ),
        # Wrong line numbers (both line 1)
        (
            "1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9997",
            "1 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110516"
        ),
        # Invalid checksum on line 1
        (
            "1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  999",
            "2 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110516"
        ),
        # Invalid checksum on line 2
        (
            "1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9997",
            "2 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.0027772611051"
        ),
        # Wrong line numbers (line 2 marked as line 3)
        (
            "1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9997",
            "3 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110517"
        ),
        # Wrong line numbers (both line 2)
        (
            "2 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9998",
            "2 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110516"
        ),
        # Mismatched NORAD IDs (different by 1)
        (
            "1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9997",
            "2 23614  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110517"
        ),
    ])
    def test_validate_tle_lines_invalid(self, line1, line2):
        """Test validation of invalid TLE line pairs."""
        assert not brahe.validate_tle_lines(line1, line2)


class TestNORADIDParsing:
    """Test NORAD ID parsing for classic and Alpha-5 formats."""

    @pytest.mark.parametrize("id_str,expected", [
        ("25544", 25544),
        ("00001", 1),
        ("99999", 99999),
        ("    1", 1),
    ])
    def test_parse_norad_id_classic(self, id_str, expected):
        """Test parsing of classic numeric NORAD IDs."""
        assert brahe.parse_norad_id(id_str) == expected

    @pytest.mark.parametrize("id_str,expected", [
        ("A0000", 100000),
        ("A0001", 100001),
        ("A9999", 109999),
        ("B0000", 110000),
        ("Z9999", 339999),
        ("B1234", 111234),
        ("C5678", 125678),
        ("D9012", 139012),
        ("E3456", 143456),
        ("F7890", 157890),
        ("G1234", 161234),
        ("H2345", 172345),
        ("J6789", 186789),
        ("K0123", 190123),
        ("L4567", 204567),
        ("M8901", 218901),
        ("N2345", 222345),
        ("P6789", 236789),
        ("Q0123", 240123),
        ("R4567", 254567),
        ("S8901", 268901),
        ("T2345", 272345),
        ("U6789", 286789),
        ("V0123", 290123),
        ("W4567", 304567),
        ("X8901", 318901),
        ("Y2345", 322345),
        ("Z6789", 336789),
    ])
    def test_parse_norad_id_alpha5_valid(self, id_str, expected):
        """Test parsing of valid Alpha-5 NORAD IDs."""
        assert brahe.parse_norad_id(id_str) == expected

    @pytest.mark.parametrize("id_str", [
        "I0001",    # 'I' is invalid
        "O1234",    # 'O' is invalid
        "A123",     # Too short
        "A12345",   # Too long
        "1234A",    # Invalid format
        "!2345",    # Invalid character
        "",         # Empty string
        "     ",    # Only spaces
    ])
    def test_parse_norad_id_invalid(self, id_str):
        """Test parsing of invalid NORAD IDs."""
        with pytest.raises(RuntimeError):  # brahe.BraheError in Rust becomes RuntimeError in Python
            brahe.parse_norad_id(id_str)


class TestKeplerianElementsFromTLE:
    """Test extraction of Keplerian elements from TLE lines."""

    def test_keplerian_elements_from_tle(self):
        """Test extraction of Keplerian elements from TLE."""
        line1 = "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997"
        line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"

        epoch, elements = brahe.keplerian_elements_from_tle(line1, line2)

        # Check epoch
        assert epoch.year() == 2021
        assert epoch.month() == 1
        assert epoch.day() == 1
        assert epoch.hour() == 12
        assert epoch.minute() == 0
        assert abs(epoch.second() - 0.0) < 1e-6

        # Check elements
        # Convert mean motion to semi-major axis for comparison
        n_rad_per_sec = 15.48919103 * 2.0 * np.pi / 86400.0
        expected_a = brahe.semimajor_axis(n_rad_per_sec, brahe.AngleFormat.RADIANS)

        assert abs(elements[0] - expected_a) < 1.0e-3  # Semi-major axis in meters
        assert abs(elements[1] - 0.0003417) < 1.0e-7   # Eccentricity
        assert abs(elements[2] - 51.6461) < 1.0e-4     # Inclination (degrees)
        assert abs(elements[3] - 306.0234) < 1.0e-4    # RAAN (degrees)
        assert abs(elements[4] - 88.1267) < 1.0e-4     # Argument of periapsis (degrees)
        assert abs(elements[5] - 25.5695) < 1.0e-4     # Mean anomaly (degrees)


class TestCreateTLELines:
    """Test creation of TLE lines from orbital elements."""

    def test_create_tle_lines(self):
        """Test creation of TLE lines with various parameters."""
        epoch = brahe.Epoch.from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
        semi_major_axis = 6786000.0  # meters
        eccentricity = 0.12345      # dimensionless
        inclination = 51.6461       # degrees
        raan = 306.0234            # degrees
        arg_periapsis = 88.1267    # degrees
        mean_anomaly = 25.5695     # degrees

        # Convert semi-major axis to mean motion (rev/day)
        mean_motion_rad_per_sec = (brahe.GM_EARTH / (semi_major_axis ** 3)) ** 0.5
        mean_motion_revs_per_day = mean_motion_rad_per_sec * 86400.0 / (2.0 * np.pi)

        # Test with negative derivatives
        line1, line2 = brahe.create_tle_lines(
            epoch,
            inclination,
            raan,
            eccentricity,
            arg_periapsis,
            mean_anomaly,
            mean_motion_revs_per_day,
            "25544",            # norad_id
            0,                  # ephemeris_type
            999,                # element_set_number
            12345,              # revolution_number
            classification='U',
            intl_designator="98067A",
            first_derivative=-0.00001764,
            second_derivative=-0.00000067899,
            bstar=-0.00012345,
        )

        expected_line1 = "1 25544U 98067A   21001.50000000 -.00001764 -67899-7 -12345-4 0 09995"
        expected_line2 = "2 25544  51.6461 306.0234 1234500  88.1267  25.5695 15.53037630123450"

        assert line1 == expected_line1
        assert line2 == expected_line2

        # Test with positive derivatives
        line1_pos, line2_pos = brahe.create_tle_lines(
            epoch,
            inclination,
            raan,
            eccentricity,
            arg_periapsis,
            mean_anomaly,
            mean_motion_revs_per_day,
            "25544",            # norad_id
            0,                  # ephemeris_type
            999,                # element_set_number
            12345,              # revolution_number
            classification='U',
            intl_designator="98067A",
            first_derivative=0.00001764,
            second_derivative=0.00000067899,
            bstar=0.00012345,
        )

        expected_line1_pos = "1 25544U 98067A   21001.50000000  .00001764  67899-7  12345-4 0 09992"
        expected_line2_pos = "2 25544  51.6461 306.0234 1234500  88.1267  25.5695 15.53037630123450"

        assert line1_pos == expected_line1_pos
        assert line2_pos == expected_line2_pos


class TestKeplerianElementsToTLE:
    """Test conversion of Keplerian elements to TLE format."""

    def test_keplerian_elements_to_tle(self):
        """Test conversion from Keplerian elements to TLE."""
        epoch = brahe.Epoch.from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
        semi_major_axis = 6786000.0  # meters
        eccentricity = 0.12345      # dimensionless
        inclination = 51.6461       # degrees
        raan = 306.0234            # degrees
        arg_periapsis = 88.1267    # degrees
        mean_anomaly = 25.5695     # degrees

        elements = np.array([
            semi_major_axis,
            eccentricity,
            inclination,
            raan,
            arg_periapsis,
            mean_anomaly,
        ])

        line1, line2 = brahe.keplerian_elements_to_tle(epoch, elements, "25544")

        expected_line1 = "1 25544U          21001.50000000  .00000000  00000+0  00000+0 0 00000"
        expected_line2 = "2 25544  51.6461 306.0234 1234500  88.1267  25.5695 15.53037630000005"

        assert line1 == expected_line1
        assert line2 == expected_line2


class TestNORADIDConversions:
    """Test NORAD ID conversion functions."""

    @pytest.mark.parametrize("norad_id,expected", [
        (100000, "A0000"),
        (100001, "A0001"),
        (109999, "A9999"),
        (110000, "B0000"),
        (111234, "B1234"),
        (125678, "C5678"),
        (186789, "J6789"),  # Skip I
        (236789, "P6789"),  # Skip O
        (339999, "Z9999"),
    ])
    def test_norad_id_numeric_to_alpha5_valid(self, norad_id, expected):
        """Test conversion of valid numeric NORAD IDs to Alpha-5 format."""
        assert brahe.norad_id_numeric_to_alpha5(norad_id) == expected

    @pytest.mark.parametrize("norad_id", [
        99999,    # Too low
        340000,   # Too high
        0,        # Way too low
        999999,   # Way too high
    ])
    def test_norad_id_numeric_to_alpha5_invalid(self, norad_id):
        """Test conversion of invalid numeric NORAD IDs."""
        with pytest.raises(RuntimeError):  # brahe.BraheError in Rust becomes RuntimeError in Python
            brahe.norad_id_numeric_to_alpha5(norad_id)

    @pytest.mark.parametrize("norad_id", [
        100000, 100001, 109999, 110000, 125678, 186789, 236789, 339999
    ])
    def test_norad_id_alpha5_numeric_round_trip(self, norad_id):
        """Test round trip conversion from numeric to Alpha-5 and back."""
        alpha5 = brahe.norad_id_numeric_to_alpha5(norad_id)
        parsed_id = brahe.parse_norad_id(alpha5)
        assert norad_id == parsed_id, f"Round trip failed for ID {norad_id}: {alpha5} -> {parsed_id}"


class TestTLECircularity:
    """Test circularity between TLE and Keplerian element conversions."""

    def test_keplerian_tle_circularity(self):
        """Test circularity: Keplerian elements -> TLE -> Keplerian elements."""
        # Original Keplerian elements
        original_epoch = brahe.Epoch.from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
        original_elements = np.array([
            6786000.0,  # Semi-major axis (m)
            0.12345,    # Eccentricity
            51.6461,    # Inclination (degrees)
            306.0234,   # RAAN (degrees)
            88.1267,    # Argument of periapsis (degrees)
            25.5695,    # Mean anomaly (degrees)
        ])
        norad_id = "25544"

        # Convert Keplerian elements to TLE
        line1, line2 = brahe.keplerian_elements_to_tle(original_epoch, original_elements, norad_id)

        # Convert TLE back to Keplerian elements
        recovered_epoch, recovered_elements = brahe.keplerian_elements_from_tle(line1, line2)

        # Check that epoch matches (within reasonable precision)
        assert recovered_epoch.year() == original_epoch.year()
        assert recovered_epoch.month() == original_epoch.month()
        assert recovered_epoch.day() == original_epoch.day()
        assert recovered_epoch.hour() == original_epoch.hour()
        assert recovered_epoch.minute() == original_epoch.minute()
        assert abs(recovered_epoch.second() - original_epoch.second()) < 1e-6

        # Check that elements match (within reasonable precision for TLE format limitations)
        assert abs(recovered_elements[0] - original_elements[0]) < 1.0      # Semi-major axis (m)
        assert abs(recovered_elements[1] - original_elements[1]) < 1e-6     # Eccentricity
        assert abs(recovered_elements[2] - original_elements[2]) < 1e-3     # Inclination (degrees)
        assert abs(recovered_elements[3] - original_elements[3]) < 1e-3     # RAAN (degrees)
        assert abs(recovered_elements[4] - original_elements[4]) < 1e-3     # Argument of periapsis (degrees)
        assert abs(recovered_elements[5] - original_elements[5]) < 1e-3     # Mean anomaly (degrees)

    def test_tle_keplerian_circularity(self):
        """Test circularity: TLE -> Keplerian elements -> TLE."""
        # Original TLE
        original_line1 = "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997"
        original_line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"

        # Convert TLE to Keplerian elements
        epoch, elements = brahe.keplerian_elements_from_tle(original_line1, original_line2)

        # Convert back to TLE
        recovered_line1, recovered_line2 = brahe.keplerian_elements_to_tle(epoch, elements, "25544")

        # Extract NORAD ID from both TLEs to compare (columns 2-6 of line 1)
        original_norad_id = original_line1[2:7].strip()
        recovered_norad_id = recovered_line1[2:7].strip()
        assert original_norad_id == recovered_norad_id

        # Parse both TLEs and compare elements (since exact string match is not expected due to formatting differences)
        _, original_elements = brahe.keplerian_elements_from_tle(original_line1, original_line2)
        _, recovered_elements = brahe.keplerian_elements_from_tle(recovered_line1, recovered_line2)

        # Elements should match within TLE precision limits
        assert abs(recovered_elements[0] - original_elements[0]) < 1.0      # Semi-major axis (m)
        assert abs(recovered_elements[1] - original_elements[1]) < 1e-6     # Eccentricity
        assert abs(recovered_elements[2] - original_elements[2]) < 1e-3     # Inclination (degrees)
        assert abs(recovered_elements[3] - original_elements[3]) < 1e-3     # RAAN (degrees)
        assert abs(recovered_elements[4] - original_elements[4]) < 1e-3     # Argument of periapsis (degrees)
        assert abs(recovered_elements[5] - original_elements[5]) < 1e-3     # Mean anomaly (degrees)


