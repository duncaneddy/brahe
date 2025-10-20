# /// script
# dependencies = ["brahe"]
# ///
"""
Combining multiple groundstation networks.

Demonstrates:
- Loading multiple providers
- Merging networks into a single list
- Using combined networks for analysis
"""

import brahe as bh

if __name__ == "__main__":
    print("Combining Groundstation Networks")
    print("=" * 60)

    # Load multiple providers
    primary = bh.datasets.groundstations.load("ksat")
    backup = bh.datasets.groundstations.load("ssc")

    print(f"\nPrimary network (KSAT): {len(primary)} stations")
    print(f"Backup network (SSC): {len(backup)} stations")

    # Combine into single network
    combined = primary + backup

    print(f"Combined network: {len(combined)} stations")

    # Analyze combined coverage
    arctic = [s for s in combined if s.lat > 66.5]
    print(f"Arctic coverage: {len(arctic)} stations")

    print("\n" + "=" * 60)
    print("Network combination complete!")
