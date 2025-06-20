import pandas as pd
import numpy as np
import geopandas as gpd
import h3pandas
from typing import cast
import h3


def create_hex(aoi: gpd.GeoDataFrame, resolution: int) -> gpd.GeoDataFrame:
    geom = aoi.h3.polyfill_resample(resolution).get(["geometry"])
    geom = geom.assign(h3_index=geom.index)
    geom = geom.reset_index(drop=True)
    return geom


# def add_neighbors(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
#     orig_crs = gdf.crs
#     gdf = cast(gpd.GeoDataFrame, gdf.to_crs(epsg=3857))
#     nei = []
#     for idx, row in gdf.iterrows():
#         out = (
#             gpd.GeoDataFrame(geometry=[row.geometry], crs=gdf.crs)
#             .sjoin_nearest(gdf, how="left", max_distance=0.1)
#             .index_right
#         )
#         out = out.loc[out != idx]
#         out = np.pad(out, (0, 6 - len(out)))
#         nei.append(out)
#
#     nei = pd.DataFrame(nei, columns=[f"n{i}" for i in range(6)], index=gdf.index)  # type: ignore
#     gdf = pd.concat((gdf, nei), axis=1).to_crs(orig_crs)
#     return gdf

def add_neighbors(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    This function has been completely rewritten for performance.

    Instead of a slow, iterative spatial join for each hexagon, this version uses the
    efficient algorithmic neighbor-finding capabilities of the H3 library.
    """
    # Create a mapping from the h3_index to the DataFrame's integer index.
    # This allows us to quickly find the integer index of any neighboring hexagon.
    h3_to_int_index = pd.Series(gdf.index, index=gdf['h3_index'])

    # Use h3.k_ring to get the neighbors for each hexagon's H3 index.
    # We apply a function to the 'h3_index' column. This is vectorized and fast.
    # For each h3_address:
    # 1. Get all neighbors in ring 1 (including the center hexagon) with h3.k_ring.
    # 2. Remove the original h3_address from the set of neighbors.
    # 3. Map the remaining neighbor h3 addresses to their integer indices using our lookup series.
    # 4. Convert to a list and pad with None to ensure all lists have length 6.


    def get_neighbor_indices(h3_address):
        # Get neighbors (including self)
        neighbors_h3 = h3.k_ring(h3_address, 1)
        # Remove self
        neighbors_h3.remove(h3_address)

        # --- THIS IS THE CORRECTED PART ---
        # Use .reindex() to look up the integer indices for the neighbor H3 addresses.
        # This correctly handles cases where a neighbor is not in our gdf (e.g., at the border),
        # by returning NaN for those missing neighbors.
        neighbor_series = h3_to_int_index.reindex(list(neighbors_h3))

        # Now, drop the NaNs (for neighbors outside our area) and get the list of valid integer indices.
        neighbor_int_indices = neighbor_series.dropna().astype(int).tolist()

        # Pad the list to ensure it has 6 elements, matching the original output format
        padded_indices = neighbor_int_indices + [np.nan] * (6 - len(neighbor_int_indices))
        return padded_indices

    neighbor_indices = gdf['h3_index'].apply(get_neighbor_indices)

    # Convert the list of lists into a DataFrame
    nei_df = pd.DataFrame(
        neighbor_indices.tolist(),
        columns=[f"n{i}" for i in range(6)],
        index=gdf.index
    )

    # Concatenate the neighbor DataFrame with the original GeoDataFrame
    gdf_with_neighbors = pd.concat([gdf, nei_df], axis=1)

    return gdf_with_neighbors


def feat(aoi: gpd.GeoDataFrame, hex_res: int) -> gpd.GeoDataFrame:
    geom = create_hex(aoi, hex_res)
    geom = add_neighbors(geom)
    geom["index"] = geom.index
    geom = cast(gpd.GeoDataFrame, geom.dropna(axis=0, subset=["geometry"]))
    return geom

