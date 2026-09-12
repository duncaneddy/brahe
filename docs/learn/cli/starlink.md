# Starlink Commands

The `starlink` command group lists and downloads Starlink's public Modified ITC ephemerides. Downloads are cached under `$BRAHE_CACHE/starlink/` and reused on subsequent invocations. Setting `BRAHE_NETWORK_MODE=offline` serves cached data without any request; see [Environment Variables](../utilities/environment_variables.md).

## Commands

### `manifest`

List the satellites in the Starlink manifest.

**Syntax:**
```bash
brahe starlink manifest [OPTIONS]
```

**Options:**
- `--norad-id <id>` - Show only this NORAD catalog number
- `--name <name>` - Show only this object name (exact match)
- `--limit <n>` - Show at most this many entries

**Examples:**

List the first entries in the manifest:
```bash
brahe starlink manifest --limit 3
```
Output:
```
                                                                         Starlink manifest (3 of 5 entries)
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ NORAD ID ┃ Object         ┃ Category    ┃ Ephemeris start             ┃ Ephemeris stop              ┃ File                                                                       ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│   100001 │ STARLINK-38128 │ Operational │ 2026-09-11 01:42:00.000 UTC │ 2026-09-14 01:43:00.000 GPS │ MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt │
│   100002 │ STARLINK-37711 │ Operational │ 2026-09-11 01:49:00.000 UTC │ 2026-09-14 01:50:00.000 GPS │ MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt │
│   100003 │ STARLINK-38123 │ Operational │ 2026-09-11 01:41:00.000 UTC │ 2026-09-14 01:42:00.000 GPS │ MEME_100003_STARLINK-38123_2540141_Operational_1473385320_UNCLASSIFIED.txt │
└──────────┴────────────────┴─────────────┴─────────────────────────────┴─────────────────────────────┴────────────────────────────────────────────────────────────────────────────┘
```

Look up a single satellite by NORAD ID:
```bash
brahe starlink manifest --norad-id 100001
```

Look up a single satellite by object name:
```bash
brahe starlink manifest --name STARLINK-38128
```

An unknown `--norad-id` or `--name` exits with an error and does not print a table.

---

### `download`

Download ephemeris files for one or more satellites.

**Syntax:**
```bash
brahe starlink download NORAD_ID... [OPTIONS]
```

**Arguments:**
- `NORAD_ID...` - One or more NORAD catalog numbers

**Options:**
- `--output`, `-o <dir>` - Directory to copy the files into; the cache is used when omitted

**Examples:**

Download into the cache:
```bash
brahe starlink download 100002
```
Output:
```
Downloaded 100002 to /Users/duncan/.cache/brahe/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt
```

Download and copy into a directory:
```bash
brahe starlink download 100002 --output ./ephemerides
```
Output:
```
Downloaded 100002 to ephemerides/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt
```

Download multiple satellites in one call:
```bash
brahe starlink download 100001 100002 100003
```

---

## See Also

- [Ephemeris Data Sources](../ephemeris/index.md) - CelesTrak, Space-Track, and Starlink ephemeris sources
- [Datasets CLI](datasets.md) - Other satellite dataset downloads
- [CLI API](../../library_api/index.md) - Python API documentation
