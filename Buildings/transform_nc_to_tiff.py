import rasterio
import config
import geopandas as gpd
from pathlib import Path
import rasterio.mask

# --- Path Setup ---
# This makes the script runnable from any directory
try:
    # Assumes the script is in a subdirectory of 'EDeMOS'
    current_script_path = Path(__file__).resolve()
    # Find the 'EDeMOS' root directory
    ROOT_DIR = next(p for p in current_script_path.parents if p.name == 'EDeMOS')
except (NameError, StopIteration):
    # Fallback for interactive use (like Jupyter notebooks) or if structure is different
    # Assumes the current working directory is 'EDeMOS' or a subdir
    ROOT_DIR = Path.cwd()
    while ROOT_DIR.name != 'EDeMOS' and ROOT_DIR.parent != ROOT_DIR:
        ROOT_DIR = ROOT_DIR.parent
    print(f"Running in interactive mode. Set ROOT_DIR to: {ROOT_DIR}")



def nc_to_tif_and_crop(nc_file_path, base_tif_path, crop_gdf, country_name):
    """
    Converts NetCDF bands to separate, cropped GeoTIFF files.

    Args:
        nc_file_path (pathlib.Path): Path to the input NetCDF file.
        base_tif_path (pathlib.Path): Base path for the output GeoTIFFs.
        crop_gdf (geopandas.GeoDataFrame): GeoDataFrame with geometries to crop to.
    """
    # Ensure the output directory exists
    base_tif_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(nc_file_path) as src:
        # Get number of bands in the opened dataset
        num_bands = src.count
        print(f"Found {num_bands} band(s) in {nc_file_path.name}")

        # Try to get the CRS from the file
        crs = src.crs
        if not crs:
            # If no CRS is found, default to WGS84 and print a warning
            crs = 'EPSG:4326'  # WGS84
            print("WARNING: CRS not found in source file. Assuming EPSG:4326.")

        # Ensure the vector's CRS matches the raster's CRS
        if crop_gdf.crs != crs:
            print(f"Warning: Reprojecting crop geometry from {crop_gdf.crs} to {crs}")
            crop_gdf = crop_gdf.to_crs(src.crs)

        # Get the geometries for masking
        geoms = crop_gdf.geometry.values

        # --- Perform the crop operation using rasterio.mask ---
        # This reads the data and crops it in one go.
        # It returns the cropped data array and the new transform for the cropped extent.
        try:
            cropped_data, cropped_transform = rasterio.mask.mask(src, geoms, crop=True)
            # Get the new height and width from the cropped data array
            out_meta = src.meta.copy()
            nodata_val = src.nodata
        except ValueError as e:
            print(f"Error during cropping: {e}")
            print("This often means the crop geometry does not overlap with the raster.")
            return

        # Update the metadata with the new dimensions, transform, and other properties
        out_meta.update({
            "driver": "GTiff",
            "height": cropped_data.shape[1],
            "width": cropped_data.shape[2],
            "transform": cropped_transform,
            "nodata": nodata_val
        })

        # The cropped_data array has a shape of (num_bands, height, width)
        num_bands = cropped_data.shape[0]
        print(f"Found and cropped {num_bands} band(s) from {nc_file_path.name}")

        for band_idx in range(num_bands):
            # The band index for rasterio is 1-based, array index is 0-based
            current_band_num = band_idx + 1

            tif_file = base_tif_path.parent / f"{base_tif_path.name}_band{current_band_num}_{country_name}.tif"

            # Update metadata for single-band output
            single_band_meta = out_meta.copy()
            single_band_meta['count'] = 1
            single_band_meta['dtype'] = cropped_data.dtype

            with rasterio.open(tif_file, "w", **single_band_meta) as dst:
                # Write the specific band from the cropped data array
                dst.write(cropped_data[band_idx], 1)

            print(f"  -> Saved cropped band {current_band_num} to {tif_file.name}")


if __name__ == '__main__':
    # Define input and output paths using the robust ROOT_DIR
    in_nc_file = ROOT_DIR / "Buildings/Data/Falchetta_ElecAccess/tiersofaccess_SSA_2018.nc"
    out_base_tif = ROOT_DIR / "Buildings/Data/Falchetta_ElecAccess/tiersofaccess_SSA_2018"

    admin_gpkg_path = config.ADMIN_PATH / config.ADMIN_GPKG
    admin_gdf = gpd.read_file(admin_gpkg_path, layer=config.ADMIN_LAYER_COUNTRY)

    if not in_nc_file.exists():
        print(f"Error: Input file not found at {in_nc_file}")
    else:
        nc_to_tif_and_crop(in_nc_file, out_base_tif, admin_gdf, config.COUNTRY)
        print(f"\nConversion complete for all bands from {in_nc_file.name}")