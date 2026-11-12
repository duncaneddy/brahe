# /// script
# dependencies = ["brahe"]
# FLAGS = ["NETWORK"]
# ///
"""
List the IGS precise products published for one GPS week and filter them by center, solution, and content.

Uses the Garner (SIO) mirror, which needs no credentials. Garner is served over
plain HTTP, so the downloaded files are not transport-secured.
"""

import brahe as bh

client = bh.IGSProductsClient(bh.IGSProductSource.GARNER)
print(f"Source: {client.source}, base URL: {client.base_url}")

# Any epoch inside the week selects it; this one falls in GPS week 2340
epoch = bh.Epoch(2024, 11, 12, 12, 0, 0.0, time_system=bh.GPS)
print(f"Files published for the week: {len(client.list_files(epoch))}")

# Final orbit products in SP3 format from the CODE and IGS analysis centers
for center in (bh.IGSAnalysisCenter.COD, bh.IGSAnalysisCenter.IGS):
    products = client.list_products(
        epoch,
        center=center,
        solution=bh.IGSSolutionType.FINAL,
        content=bh.IGSContentType.ORBIT,
    )
    print(f"\n{center} final SP3 orbits: {len(products)}")
    for product in products[:3]:
        name = product.file_name
        print(
            f"  {product.name}: starts {name.start_epoch}, "
            f"span {name.span / 3600.0:.0f} h, sampling {name.sampling:.0f} s"
        )
