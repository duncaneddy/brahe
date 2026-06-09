# /// script
# dependencies = ["brahe"]

import brahe as bh

## Static Providers

# The simplest data providers are static providers. These always return the same value regarless of date.
static_provider = bh.StaticEOPProvider.from_zero()  # All values are zero

# Now we set the global provider as the static provider, which will be used for all EOP data access across the library.
bh.set_global_eop_provider(static_provider)

# Similarly we can use a file provider, which loads data from a local file
# The from_default_standard loads the IERS Standard EOP product that is bundled with the package
# The first argument (True) says to inerpolate values for times between data points
# The second argument ("Hold") says to hold the last value for times outside the range of the data, instead of throwing an error
default_file_provider = bh.FileEOPProvider.from_default_standard(True, "Hold")
bh.set_global_eop_provider(default_file_provider)

# There are also functions to explicitly download a specific file and and load it from a file
eop_path = "eop_c04.txt"
bh.download_c04_eop_file(eop_path)
c04_file_provider = bh.FileEOPProvider.from_c04_file(eop_path, True, "Hold")
bh.set_global_eop_provider(c04_file_provider)
print("EOP data available through:", bh.Epoch(bh.get_global_eop_mjd_max()))

eop_path = "eop_standard.txt"
bh.download_standard_eop_file(eop_path)
standard_file_provider = bh.FileEOPProvider.from_standard_file(eop_path, True, "Hold")
bh.set_global_eop_provider(standard_file_provider)
print("EOP data available through:", bh.Epoch(bh.get_global_eop_mjd_max()))