# Record Classes

SpaceTrack API queries return typed record objects. Each record class provides:

- **Property access** - Access fields as Python properties (e.g., `record.norad_cat_id`)
- **`as_dict()` method** - Convert to dictionary with snake_case keys
- **`__repr__()` method** - Human-readable string representation

## BasicSpaceData Records

### GPRecord

General Perturbations orbital elements, the primary source for TLE/orbital data.

::: brahe.GPRecord

---

### GPHistoryRecord

Historical GP element sets for tracking orbital evolution over time.

::: brahe.GPHistoryRecord

---

### OMMRecord

Orbit Mean-elements Message format records.

::: brahe.OMMRecord

---

### TLERecord

Two-Line Element records (deprecated, use GPRecord instead).

::: brahe.TLERecord

---

### SATCATRecord

Satellite Catalog entries with object metadata.

::: brahe.SATCATRecord

---

### SATCATChangeRecord

Records of changes to satellite catalog entries.

::: brahe.SATCATChangeRecord

---

### SATCATDebutRecord

New objects added to the satellite catalog.

::: brahe.SATCATDebutRecord

---

### DecayRecord

Predicted and actual re-entry information.

::: brahe.DecayRecord

---

### TIPRecord

Tracking and Impact Prediction messages for decaying objects.

::: brahe.TIPRecord

---

### CDMPublicRecord

Public Conjunction Data Messages for collision analysis.

::: brahe.CDMPublicRecord

---

### BoxscoreRecord

Catalog statistics summarized by country.

::: brahe.BoxscoreRecord

---

### LaunchSiteRecord

Launch facility information.

::: brahe.LaunchSiteRecord

---

### AnnouncementRecord

Space-Track system announcements.

::: brahe.AnnouncementRecord

---

## ExpandedSpaceData Records

These records require expanded access permissions on Space-Track.org.

### CDMRecord

Full Conjunction Data Messages (requires expanded access).

::: brahe.CDMRecord

---

### CARRecord

Conjunction Assessment Reports (requires expanded access).

::: brahe.CARRecord

---

### ManeuverRecord

Satellite maneuver data (requires expanded access).

::: brahe.ManeuverRecord

---

### ManeuverHistoryRecord

Historical satellite maneuver records (requires expanded access).

::: brahe.ManeuverHistoryRecord

---

### OrganizationRecord

Organization/operator information (requires expanded access).

::: brahe.OrganizationRecord

---

### SatelliteRecord

Detailed satellite information (requires expanded access).

::: brahe.SatelliteRecord

## See Also

- [SpaceTrackClient](client.md) - Client for querying records
- [SpaceTrack User Guide](../../learn/spacetrack/index.md) - Usage examples
