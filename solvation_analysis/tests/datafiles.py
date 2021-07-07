from pkg_resources import resource_filename
import pathlib
from pathlib import Path

bn_fec_data = resource_filename(__name__, "data/bn_fec.data")
bn_fec_dcd_unwrap = resource_filename(__name__, "data/bn_fec_short_unwrap.dcd")
bn_fec_dcd_wrap = resource_filename(__name__, "data/bn_fec_short_wrap.dcd")
bn_fec_atom_types = resource_filename(__name__, "data/bn_fec_elements.csv")

data_dir = Path("data")
easy_rdf_dir = data_dir / "rdf_vs_li_easy"
hard_rdf_dir = data_dir / "rdf_vs_li_hard"
fail_rdf_dir = data_dir / "rdf_non_solvated"


def generate_tag(filename):
    tag_list = filename.stem.split("_")
    rdf_tag = f"{tag_list[1]}_{tag_list[2]}"
    return rdf_tag


easy_rdf_bins = {
    generate_tag(rdf_path): resource_filename(__name__, str(rdf_path))
    for rdf_path in easy_rdf_dir.glob("*bins.npz")
}

easy_rdf_data = {
    generate_tag(rdf_path): resource_filename(__name__, str(rdf_path))
    for rdf_path in easy_rdf_dir.glob("*data.npz")
}

hard_rdf_bins = {
    generate_tag(rdf_path): resource_filename(__name__, str(rdf_path))
    for rdf_path in hard_rdf_dir.glob("*bins.npz")
}

hard_rdf_data = {
    generate_tag(rdf_path): resource_filename(__name__, str(rdf_path))
    for rdf_path in hard_rdf_dir.glob("*data.npz")
}

non_solv_rdf_bins = {
    generate_tag(rdf_path): resource_filename(__name__, str(rdf_path))
    for rdf_path in fail_rdf_dir.glob("*bins.npz")
}

non_solv_rdf_data = {
    generate_tag(rdf_path): resource_filename(__name__, str(rdf_path))
    for rdf_path in fail_rdf_dir.glob("*data.npz")
}

del resource_filename
