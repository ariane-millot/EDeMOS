"""File reduction for website"""

from pathlib import Path
import geopandas as gpd

DEFAULT_COLUMNS_TO_KEEP = [
    'h3_index', 'geometry',
    'n0', 'n1', 'n2', 'n3', 'n4', 'n5',
    'NAME_1',
    'locationWP',
    'HH_total',
    'pop_total',
    'HHwithAccess',
    'total_employee',
    'res_elec_kWh_meth2',
    'res_elec_kWh_meth2_scaled',
    'ser_elec_kWh_final',
    'ind_elec_GWh',
]


def reduce_gpkg_for_website(input_file, output_file, columns_to_keep=None):
    """
    Subsets columns of a total electricity consumption geopackage for web publication.
    """
    if columns_to_keep is None:
        columns_to_keep = DEFAULT_COLUMNS_TO_KEEP

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    gdf = gpd.read_file(input_path)
    existing_cols = [col for col in columns_to_keep if col in gdf.columns]
    gdf_reduced = gdf[existing_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_reduced.to_file(output_path, driver='GPKG')
    print(f"Reduced geopackage saved to {output_path}")
    return gdf_reduced


if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parent
    out_dir = root_dir / "Outputs"
    in_file = out_dir / "total_electricity_consumption_Zambia.gpkg"
    out_file = out_dir / "total_elec_consumption_web.gpkg"

    if in_file.exists():
        reduce_gpkg_for_website(in_file, out_file)
    else:
        print(f"Notice: Sample input file not found at {in_file}. Skipping execution.")