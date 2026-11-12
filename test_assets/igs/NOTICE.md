# IGS listing fixtures

The files in this directory are directory listings of IGS product archives,
used to test listing and filtering in `brahe::igs`. The products they list are
published by the International GNSS Service (IGS) and its analysis centers.

- `garner_root.html`: the Apache listing of
  `http://garner.ucsd.edu/pub/products/` (Scripps Institution of
  Oceanography), captured 2026-11-03 and trimmed to the header, sort links,
  parent link, the week directories 0576-0580, 2339-2341 and 2442-2443, the
  three-digit `860/` directory, and every non-week entry.
- `garner_2340.html`: the Apache listing of
  `http://garner.ucsd.edu/pub/products/2340/`, captured 2026-11-03 and trimmed
  to the header, sort links, parent link, every `20243150000` product, and a
  selection of other long-name, short-name, and non-product entries. The
  `readme.txt` and `repro3/` rows are not in the archive; they were added in
  the archive's row format so the fixture holds a plain file and a
  subdirectory.
- `bkg_2340.html`: the Apache listing of
  `https://igs.bkg.bund.de/root_ftp/IGS/products/2340/` (BKG), captured
  2026-11-03 and trimmed the same way, with the same two added rows.
- `cddis_2340.txt`: a hand-written CDDIS `*?list` listing of week 2340 with one
  `name size` line per file of `garner_2340.html`. Sizes are in bytes,
  converted from the Garner listing's rounded sizes.
