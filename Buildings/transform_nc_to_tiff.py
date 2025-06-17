import rasterio
from rasterio. vrt import WarpedVRT


def nc_to_tif(nc_file, base_tif_file):
  """  Converts a NetCDF file to separate GeoTIFF files for each band.

  Args:
      nc_file (str): Path to the input NetCDF file.
      base_tif_file (str): Base filename for the output GeoTIFF files (band number will be appended).
  """

  with rasterio.open(nc_file) as src:
    # Get number of bands
    num_bands = src.count

    for band in range(1, num_bands + 1):
      # Get data from the current band
      data = src.read(band)
      print(data)
      # Get geotransform information (coordinates)
      transform = src.transform
      print(transform)

      # Get information about the data type
      dtype = src.dtypes[band - 1]
      print(dtype)

      # Get CRS (coordinate reference system) if available
      # crs = src.crs
      # Specify WGS84 CRS since it's not available in the NetCDF file
      crs = 'EPSG:4326'  # WGS84
      print(crs)

      # Create filename for the band's TIFF
      tif_file = f"{base_tif_file}_band{band}.tif"

      #  Open the output GeoTIFF file
      with rasterio.open(
          tif_file, 'w', driver='GTiff', height=src.shape[0], width=src.shape[1], count=1, dtype=dtype, crs=crs, transform=transform
      ) as dst:
        # Write the data to the GeoTIFF file
        dst.write(data, 1)

      print(f"Converted band {band} to {tif_file}")

import sys
import os
currentdir = os.path.abspath(os.getcwd())
if os.path.basename(currentdir) != 'DemandMappingZambia':
  sys.path.insert(0, os.path.dirname(currentdir))
  os.chdir('..')
  print(f'Move to {os.getcwd()}')
ROOT_DIR = os.path.abspath(os.curdir)
in_path = ROOT_DIR
out_path = ROOT_DIR

# Example usage
nc_file = in_path + "/Residential/Data/GDP/GDP_PPP_30arcsec_v3.nc"
base_tif_file = out_path + "/Residential/Data/GDP/GDP_PPP_30arcsec_v3" # Base filename for TIFFs

nc_to_tif(nc_file, base_tif_file)

print(f"Converted all bands from {nc_file}")
