# /// script
# dependencies = ["brahe"]
# ///
# FLAGS = [IGNORE]
"""
Download and save CelesTrak ephemeris data.

Demonstrates:
- Downloading ephemeris to a file
"""

import brahe as bh

if __name__ == "__main__":
    print("Download CelesTrak Ephemeris")
    print("=" * 60)

    # Download and save ephemeris
    output_file = "/tmp/gnss_satellites.json"
    bh.datasets.celestrak.download_ephemeris(
        "gnss", output_file, content_format="3le", file_format="json"
    )
    print(f"\nSaved GNSS ephemeris to: {output_file}")

    print("\n" + "=" * 60)
