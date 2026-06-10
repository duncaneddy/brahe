# /// script
# dependencies = ["brahe"]

import brahe as bh

# Create an Epoch
epoch = bh.Epoch(2024, 1, 1)

# Output as iso string
print(f"Epoch as ISO 8601 string: {epoch}")

# Output as Modified Julian Date
print(f"Epoch as Modified Julian Date: {epoch.mjd()}")

# Output as calendar date
print(f"Epoch as calendar date tuple: {epoch.to_datetime()}")

# Output as UNIX timestamp
print(f"Epoch as UNIX timestamp: {epoch.unix_timestamp()}")

# Output as GPS time
print(f"Epoch as GPS time: {epoch.gps_date()}")

# Output as Python datetime
print(f"Epoch as Python datetime: {epoch.to_pydatetime()}")
