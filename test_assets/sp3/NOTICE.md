# SP3 test fixtures

## Source

The SP3 files in this directory come from the test resources of Orekit
(https://gitlab.orekit.org/orekit/orekit), directory
`src/test/resources/sp3/`. Orekit is Copyright 2002-2026 CS GROUP and is
distributed under the Apache License, Version 2.0
(https://www.apache.org/licenses/LICENSE-2.0).

Files copied unchanged from `src/test/resources/sp3/`:

- `example-a-1.sp3`, `example-a-2.sp3`, `example-c-1.sp3`, `example-c-2.sp3`,
  `example-d-1.sp3` through `example-d-5.sp3`
- `exceeded-sat-count.sp3`, `inconsistent-sampling-dates.sp3`,
  `missing-agency.sp3`, `missing-eof.sp3`, `missing-standard-deviation.sp3`,
  `three-hours.sp3`, `truncated-line.sp3`,
  `truncated-nsgf.orb.lageos2.160305.v35.sp3`, `unsupported-version.sp3`,
  `wrong-line-identifier.sp3`, `wrong-number-of-epochs.sp3`
- `too-long-comment-a.sp3`, `too-long-comment-d.sp3`,
  `too-many-comments-a.sp3`
- `wrong-clock-base-a.sp3`, `wrong-clock-base-d.sp3`,
  `wrong-pos-vel-base-a.sp3`, `wrong-pos-vel-base-d.sp3`
- `issue895-clock-record.sp3`, `issue895-header-comment.sp3`,
  `issue895-hours-increment.sp3`, `issue895-minutes-increment.sp3`,
  `issue895-no-eof.sp3`, `issue895-second-digits.sp3`,
  `issue1014-days-increment.sp3`, `issue1327-136-sats.sp3`
- `gbm19500_*.sp3`
- `gbm18432.sp3.Z`, `nsgf.orb.stella.v00.sp3.gz`

Files derived from Orekit resources:

- `esaBHN.sp3`, `esaPRO.sp3`, `unknownType.sp3`: decompressed from the
  Orekit files of the same name with a `.Z` suffix.
- `nsgf.orb.stella.v00.sp3`: decompressed from `nsgf.orb.stella.v00.sp3.gz`.
- `example-d-2-glo.sp3`: `example-d-2.sp3` with the time system field of the
  first `%c` line (columns 10-12) changed from `GPS` to `GLO`.

## Data credits

The orbit and clock data in these files are products of the International
GNSS Service (IGS), the GFZ German Research Centre for Geosciences (`gbm*`
files), the ESA/ESOC Navigation Support Office (`esa*` and `unknownType`
files), and the International Laser Ranging Service (ILRS) and its analysis
centers (`nsgf.*`, `truncated-nsgf.*`, `issue895-*`, `issue1014-*` files).
The `example-*` files reproduce the examples of the SP3 format
specifications published by the IGS and the U.S. National Geodetic Survey.
