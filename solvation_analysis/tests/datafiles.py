__all__ = ["bn_fec.data", "bn_fec_short.dcd"]

from pkg_resources import resource_filename

bn_fec_data = resource_filename(__name__, "data/bn_fec.data")
bn_fec_dcd = resource_filename(__name__, "data/bn_fec_short.dcd")

del resource_filename
