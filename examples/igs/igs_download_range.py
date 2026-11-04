# /// script
# dependencies = ["brahe"]
# FLAGS = ["NETWORK"]
# ///
"""
Download a short range of IGS final orbit products, chain them into continuous satellite trajectories, and save the files.

Uses the Garner (SIO) mirror, which needs no credentials. Garner is served over
plain HTTP, so the downloaded files are not transport-secured.
"""

import tempfile
from pathlib import Path

import brahe as bh

bh.initialize_eop()

client = bh.IGSProductsClient(source="GARNER")
start = bh.Epoch(2024, 11, 11, 0, 0, 0.0, time_system=bh.GPS)
end = bh.Epoch(2024, 11, 12, 0, 0, 0.0, time_system=bh.GPS)

# Daily CODE final orbit files covering the range, plus one day on each side
files = client.download_sp3_range(start, end, "COD", "FIN")
print(f"Downloaded {len(files)} daily SP3 files")
for sp3 in files:
    print(f"  {sp3.start_epoch} to {sp3.end_epoch}: {sp3.num_epochs} epochs")

# One continuous trajectory per satellite across all files
satellites = bh.SP3Satellite.from_files(files)
g05 = next(s for s in satellites if s.id == "G05")
print(f"\nG05: {len(g05)} samples from {g05.start_epoch} to {g05.end_epoch}")

for hours in (0.0, 6.0, 12.0, 18.0):
    epoch = start + hours * 3600.0
    state = g05.state_eci(epoch)
    print(
        f"{epoch}: r = {state[:3].round(1)} m, "
        f"|v| = {(state[3:] @ state[3:]) ** 0.5:.1f} m/s"
    )

# Save the files covering the range, decompressed, to a directory of choice
with tempfile.TemporaryDirectory() as directory:
    paths = client.save_between(
        start,
        end,
        directory,
        center="COD",
        solution=bh.IGSSolutionType.FINAL,
        content=bh.IGSContentType.ORBIT,
        format=bh.IGSFileFormat.SP3,
    )
    print(f"\nSaved {len(paths)} files:")
    for path in paths:
        print(f"  {Path(path).name}")
