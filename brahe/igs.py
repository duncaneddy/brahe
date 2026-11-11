"""
IGS Module

Names, analysis centers, and an archive client for the International GNSS
Service (IGS) precise products. IGS analysis centers publish orbit, clock, and
Earth-rotation products on public archives under two file-name conventions:
long names (``IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz``) from GPS week 2238
and short names (``igs21000.sp3.Z``) before it.

This module provides:
- IGSProductsClient: Listing, download, and SP3 parsing of products from CDDIS, Garner, or BKG, cached under ``$BRAHE_CACHE/igs``
- IGSProductSource: The archives the client reads from
- IGSProduct: A product file listed in an archive's GPS week directory
- get_igs_cache_dir: The IGS cache directory
- IGSProductFileName: Parser for long and short product file names
- IGSSolutionType, IGSContentType, IGSFileFormat, IGSCompression, IGSNameStyle: Typed fields of a product file name
- IGSAnalysisCenter: Analysis centers that publish the products

Example:
    ```python
    import brahe as bh

    name = bh.igs.IGSProductFileName.parse("COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz")
    print(name.center, name.solution, name.gps_week, name.is_sp3)
    print(name.center.name)

    client = bh.igs.IGSProductsClient("GARNER")
    epoch = bh.Epoch(2024, 11, 10, 0, 0, 0.0, 0.0, time_system=bh.GPS)
    for product in client.list_products(epoch, center="COD", solution="FIN", content="ORB"):
        print(product.name, product.url)
    start = bh.Epoch(2024, 11, 11, 0, 0, 0.0, 0.0, time_system=bh.GPS)
    end = bh.Epoch(2024, 11, 12, 0, 0, 0.0, 0.0, time_system=bh.GPS)
    files = client.download_sp3_range(start, end, "COD", bh.igs.IGSSolutionType.FINAL)
    ```
"""

from brahe._brahe import (
    IGSAnalysisCenter,
    IGSCompression,
    IGSContentType,
    IGSFileFormat,
    IGSNameStyle,
    IGSProduct,
    IGSProductFileName,
    IGSProductsClient,
    IGSProductSource,
    IGSSolutionType,
    get_igs_cache_dir,
)

__all__ = [
    "IGSAnalysisCenter",
    "IGSCompression",
    "IGSContentType",
    "IGSFileFormat",
    "IGSNameStyle",
    "IGSProduct",
    "IGSProductFileName",
    "IGSProductSource",
    "IGSProductsClient",
    "IGSSolutionType",
    "get_igs_cache_dir",
]
