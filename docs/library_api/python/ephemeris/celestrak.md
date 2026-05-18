# CelesTrak

API reference for the CelesTrak client module. All types are available via `brahe.celestrak`.

## CelestrakClient

::: brahe.celestrak.CelestrakClient
    options:
      show_root_heading: true
      show_root_full_path: false

## CelestrakQuery

::: brahe.celestrak.CelestrakQuery
    options:
      show_root_heading: true
      show_root_full_path: false

## CelestrakSATCATRecord

::: brahe.celestrak.CelestrakSATCATRecord
    options:
      show_root_heading: true
      show_root_full_path: false

## CelestrakOutputFormat

::: brahe.celestrak.CelestrakOutputFormat
    options:
      show_root_heading: true
      show_root_full_path: false

## CelestrakQueryType

::: brahe.celestrak.CelestrakQueryType
    options:
      show_root_heading: true
      show_root_full_path: false

## SupGPSource

::: brahe.celestrak.SupGPSource
    options:
      show_root_heading: true
      show_root_full_path: false

## Satellite Groups

CelesTrak organizes satellites into logical groups. Use group names with `CelestrakQuery.gp.group("name")` or `CelestrakClient.get_gp(group="name")`.

### Temporal Groups

<div class="center-table" markdown="1">

| Group | Description |
|-------|-------------|
| `active` | All active satellites |
| `last-30-days` | Recently launched satellites |
| `tle-new` | Newly added TLEs (last 15 days) |

</div>

### Communications

<div class="center-table" markdown="1">

| Group | Description |
|-------|-------------|
| `starlink` | SpaceX Starlink constellation |
| `oneweb` | OneWeb constellation |
| `kuiper` | Amazon Kuiper constellation |
| `intelsat` | Intelsat satellites |
| `eutelsat` | Eutelsat constellation |
| `orbcomm` | ORBCOMM constellation |
| `telesat` | Telesat constellation |
| `globalstar` | Globalstar constellation |
| `iridium-NEXT` | Iridium constellation |
| `qianfan` | Qianfan constellation |
| `hulianwang` | Hulianwang Digui constellation |

</div>

### Earth Observation

<div class="center-table" markdown="1">

| Group | Description |
|-------|-------------|
| `weather` | Weather satellites (NOAA, GOES, Metop, etc.) |
| `earth-resources` | Earth observation (Landsat, Sentinel, etc.) |
| `planet` | Planet Labs imaging satellites |
| `spire` | Spire Global satellites |

</div>

### Navigation

<div class="center-table" markdown="1">

| Group | Description |
|-------|-------------|
| `gnss` | All navigation satellites (GPS, GLONASS, Galileo, BeiDou, QZSS, IRNSS) |
| `gps-ops` | Operational GPS satellites only |
| `glonass-ops` | Operational GLONASS satellites only |
| `galileo` | European Galileo constellation |
| `beidou` | Chinese BeiDou/COMPASS constellation |
| `sbas` | Satellite-Based Augmentation System (WAAS/EGNOS/MSAS) |

</div>

### Scientific and Special Purpose

<div class="center-table" markdown="1">

| Group | Description |
|-------|-------------|
| `science` | Scientific research satellites |
| `noaa` | NOAA satellites |
| `stations` | Space stations (ISS, Tiangong) |
| `analyst` | Analyst satellites (tracking placeholder IDs) |
| `visual` | 100 (or so) brightest objects |
| `gpz` | Geostationary Protected Zone |
| `gpz-plus` | Geostationary Protected Zone Plus |

</div>

Group names and contents evolve as missions launch, deorbit, or change status. Visit [CelesTrak GP Element Sets](https://celestrak.org/NORAD/elements) for the current complete list.

---

## See Also

- [CelesTrak Data Source](../../learn/ephemeris/celestrak.md) -- Overview, caching, and satellite groups
- [SpaceTrack Client](spacetrack/client.md) -- Alternative GP data source with the same GPRecord type
