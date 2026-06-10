# /// script
# dependencies = ["brahe"]

import brahe as bh
from datetime import datetime

# Create from calendar date
epoch_1 = bh.Epoch(2024, 1, 1)
epoch_2 = bh.Epoch(2024, 1, 1, 0, 0, 0)

# Create from ISO 8601 string
epoch_3 = bh.Epoch("2024-01-01T00:00:00Z")

# Create from string with time system
epoch_4 = bh.Epoch("2024-01-00 00:00:00 GPS")

# Create from datetime
epoch_5 = bh.Epoch(datetime(2024, 1, 1, 0, 0, 0))

# Current instant
epoch_6 = bh.Epoch.now()