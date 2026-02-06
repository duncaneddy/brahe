# Space-Track API

[Space-Track.org](https://www.space-track.org) is the public interface to the US Space Command satellite catalog, providing authoritative orbital data, satellite metadata, conjunction assessments, and decay predictions. Brahe's spacetrack module provides a typed client and fluent query builder for accessing this data programmatically.

!!! info "Account Required"
    Space-Track.org requires a free account. Register at [https://www.space-track.org/auth/createAccount](https://www.space-track.org/auth/createAccount) to obtain credentials.

## Module Overview

The spacetrack module is organized into five components:

### Enumerations

| Type | Purpose |
|------|---------|
| `RequestController` | API endpoint namespace (`BasicSpaceData`, `ExpandedSpaceData`, `FileShare`, `SPEphemeris`, `PublicFiles`) |
| `RequestClass` | Data category to query (`GP`, `SATCAT`, `Decay`, `TIP`, `CDMPublic`, etc.) |
| `SortOrder` | Result ordering direction (`Asc`, `Desc`) |
| `OutputFormat` | Response format (`JSON`, `TLE`, `CSV`, `XML`, `KVN`, etc.) |

Each `RequestClass` has a default controller. For example, `GP` and `SATCAT` use `BasicSpaceData`, while `CDMPublic` uses `ExpandedSpaceData`. The query builder selects the correct controller automatically.

### Query Builder

`SpaceTrackQuery` constructs API queries using a fluent builder pattern. All builder methods return a new query instance, allowing method chaining:

| Method | Purpose |
|--------|---------|
| `filter(field, value)` | Add a field/value filter predicate |
| `order_by(field, order)` | Sort results by field |
| `limit(count)` | Limit number of results |
| `limit_offset(count, offset)` | Paginate with limit and offset |
| `format(fmt)` | Set output format (default: JSON) |
| `predicates_filter(fields)` | Select specific fields to return |
| `metadata(enabled)` | Include query metadata in response |
| `distinct(enabled)` | Remove duplicate records |
| `empty_result(enabled)` | Return empty set instead of error when no results |
| `favorites(id)` | Filter by favorites list |
| `controller(ctrl)` | Override the default controller |
| `build()` | Produce the URL path string |

### Configuration

| Type | Purpose |
|------|---------|
| `RateLimitConfig` | Rate limit thresholds for per-minute and per-hour request windows |

### Client

`SpaceTrackClient` handles authentication and HTTP communication with Space-Track.org:

| Method | Purpose |
|--------|---------|
| `authenticate()` | Explicitly authenticate (establishes session) |
| `query_raw(query)` | Execute query, return raw response string |
| `query_json(query)` | Execute query, parse response as JSON array |
| `query_gp(query)` | Execute query, parse response as `GPRecord` list |
| `query_satcat(query)` | Execute query, parse response as `SATCATRecord` list |
| `fileshare_upload(folder_id, file_name, file_data)` | Upload a file to the file share |
| `fileshare_download(file_id)` | Download a file from the file share |
| `fileshare_download_folder(folder_id)` | Download all files in a folder (zip archive) |
| `fileshare_list_files()` | List files in the file share |
| `fileshare_list_folders()` | List folders in the file share |
| `fileshare_delete(file_id)` | Delete a file from the file share |
| `spephemeris_download(file_id)` | Download an SP ephemeris file |
| `spephemeris_list_files()` | List available SP ephemeris files |
| `spephemeris_file_history()` | List SP ephemeris file history |
| `publicfiles_download(file_name)` | Download a public file (no auth required) |
| `publicfiles_list_dirs()` | List public file directories (no auth required) |

Authentication is lazy by default -- the client authenticates on the first query if `authenticate()` has not been called explicitly.

### Response Types

- **`GPRecord`** -- Shared GP data record. See [Ephemeris Data Sources](../index.md) for field details.

- **`SATCATRecord`** -- Satellite Catalog record with 24 fields including object identification (`norad_cat_id`, `satname`, `intldes`), launch information (`launch`, `site`, `launch_year`), and orbital characteristics (`period`, `inclination`, `apogee`, `perigee`). `norad_cat_id` is `Optional[int]` / `Option<u32>`; remaining fields are `Optional[str]` / `Option<String>`.

- **`FileShareFileRecord`** -- File share file metadata (7 fields). All `Optional[str]` / `Option<String>`.

- **`FolderRecord`** -- File share folder metadata (4 fields). All `Optional[str]` / `Option<String>`.

- **`SPEphemerisFileRecord`** -- SP ephemeris file metadata (8 fields). `norad_cat_id` is `Optional[int]` / `Option<u32>`; remaining fields are `Optional[str]` / `Option<String>`.

### Operator Functions

Operator functions generate filter value strings for `SpaceTrackQuery.filter()`. See [Ephemeris Data Sources](../index.md) for the full operator table.

## Subpages

- [Query Builder](query_builder.md) -- Building queries with filters, ordering, and output formats
- [Client](client.md) -- Authentication, query execution, and response handling
- [File Operations](file_operations.md) -- FileShare, SP Ephemeris, and Public Files
- [Rate Limiting](rate_limiting.md) -- Configuring request rate limits

---

## See Also

- [Space-Track API Reference](../../../library_api/ephemeris/spacetrack/index.md) -- Complete function documentation
- [CelesTrak Data Source](../celestrak.md) -- Alternative TLE data source (no account required)
