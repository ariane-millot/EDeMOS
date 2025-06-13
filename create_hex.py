import pandas as pd
import numpy as np
import geopandas as gpd
import h3pandas
from typing import cast


def create_hex(aoi: gpd.GeoDataFrame, resolution: int) -> gpd.GeoDataFrame:
    geom = aoi.h3.polyfill_resample(resolution).get(["geometry"])
    geom = geom.assign(h3_index=geom.index)
    geom = geom.reset_index(drop=True)
    return geom


def add_neighbors(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    orig_crs = gdf.crs
    gdf = cast(gpd.GeoDataFrame, gdf.to_crs(epsg=3857))
    nei = []
    for idx, row in gdf.iterrows():
        out = (
            gpd.GeoDataFrame(geometry=[row.geometry], crs=gdf.crs)
            .sjoin_nearest(gdf, how="left", max_distance=0.1)
            .index_right
        )
        out = out.loc[out != idx]
        out = np.pad(out, (0, 6 - len(out)))
        nei.append(out)

    nei = pd.DataFrame(nei, columns=[f"n{i}" for i in range(6)], index=gdf.index)  # type: ignore
    gdf = pd.concat((gdf, nei), axis=1).to_crs(orig_crs)
    return gdf


def feat(aoi: gpd.GeoDataFrame, hex_res: int) -> gpd.GeoDataFrame:
    geom = create_hex(aoi, hex_res)
    geom = add_neighbors(geom)
    geom["index"] = geom.index
    geom = cast(gpd.GeoDataFrame, geom.dropna(axis=0, subset=["geometry"]))
    return geom