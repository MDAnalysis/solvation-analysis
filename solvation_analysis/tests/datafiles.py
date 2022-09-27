from pkg_resources import resource_filename
import pathlib
from pathlib import Path

bn_fec_data = resource_filename(__name__, "data/bn_fec_data/bn_fec.data")
bn_fec_dcd_unwrap = resource_filename(__name__, "data/bn_fec_data/bn_fec_short_unwrap.dcd")
bn_fec_dcd_wrap = resource_filename(__name__, "data/bn_fec_data/bn_fec_short_wrap.dcd")
bn_fec_atom_types = resource_filename(__name__, "data/bn_fec_data/bn_fec_elements.csv")
bn_fec_solv_df_large = resource_filename(__name__, "data/bn_fec_data/bn_solv_df_large.csv")
ea_fec_dcd = resource_filename(__name__, "data/ea_fec_data/ea_fec.dcd")
ea_fec_pdb = resource_filename(__name__, "data/ea_fec_data/ea_fec.pdb")
eax_data = resource_filename(__name__, "data/eax_data/")
iba_data = resource_filename(__name__, "data/iba_data/isobutyric_acid.pdb")
iba_dcd = resource_filename(__name__, "data/iba_data/isobutyric_acid.dcd")

test_dir = Path(__file__).parent
data_dir = test_dir / "data"
easy_rdf_dir = data_dir / "rdf_vs_li_easy"
hard_rdf_dir = data_dir / "rdf_vs_li_hard"
fail_rdf_dir = data_dir / "rdf_non_solvated"


def generate_short_tag(filename):
    # generates an ad-hoc tag for the li-ion rdf data from the file name
    tag_list = filename.stem.split("_")
    rdf_tag = f"{tag_list[1]}_{tag_list[2]}"
    return rdf_tag


easy_rdf_bins = {
    generate_short_tag(rdf_path): resource_filename(
        __name__, str(rdf_path.relative_to(test_dir))
    )
    for rdf_path in easy_rdf_dir.glob("*bins.npz")
}

easy_rdf_data = {
    generate_short_tag(rdf_path): resource_filename(
        __name__, str(rdf_path.relative_to(test_dir))
    )
    for rdf_path in easy_rdf_dir.glob("*data.npz")
}

hard_rdf_bins = {
    generate_short_tag(rdf_path): resource_filename(__name__, str(rdf_path.relative_to(test_dir)))
    for rdf_path in hard_rdf_dir.glob("*bins.npz")
}

hard_rdf_data = {
    generate_short_tag(rdf_path): resource_filename(__name__, str(rdf_path.relative_to(test_dir)))
    for rdf_path in hard_rdf_dir.glob("*data.npz")
}


def generate_long_tag(filename):
    # generates an ad-hoc tag for the non-solv rdf data from the file name
    tag_list = filename.stem.split("_")
    rdf_tag = f"{'_'.join(tag_list[1:3])}_{'_'.join(tag_list[4:6])}"
    return rdf_tag


non_solv_rdf_bins = {
    generate_long_tag(rdf_path): resource_filename(__name__, str(rdf_path.relative_to(test_dir)))
    for rdf_path in fail_rdf_dir.glob("*bins.npz")
}

non_solv_rdf_data = {
    generate_long_tag(rdf_path): resource_filename(__name__, str(rdf_path.relative_to(test_dir)))
    for rdf_path in fail_rdf_dir.glob("*data.npz")
}

del resource_filename
