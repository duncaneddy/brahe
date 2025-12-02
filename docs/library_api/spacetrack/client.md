# SpaceTrackClient

The `SpaceTrackClient` class provides authenticated access to the Space-Track.org API. It handles authentication, session management, rate limiting, and data retrieval.

::: brahe.SpaceTrackClient
    options:
      members:
        - __init__
        - is_authenticated
        - base_url
        - gp
        - gp_as_propagators
        - satcat
        - tle
        - decay
        - tip
        - cdm_public
        - boxscore
        - launch_site
        - satcat_change
        - satcat_debut
        - announcement
        - generic_request

## See Also

- [Record Classes](records.md) - All record types returned by client methods
- [Query Filters](../../learn/spacetrack/queries.md) - Query operators and filtering syntax
