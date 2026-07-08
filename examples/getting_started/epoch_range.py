# /// script
# dependencies = ["brahe"]

import brahe as bh

# Initialize EOP
bh.initialize_eop()

# Get epochs in a range every 6 hours
# Just like python's range it is inclusive of the start and exclusive of the end
for epoch in bh.TimeRange(bh.Epoch(2024, 1, 1), bh.Epoch(2024, 1, 2), 6 * 3600):
    print(epoch)
