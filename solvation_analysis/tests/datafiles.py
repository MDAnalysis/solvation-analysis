__all__ = ["bn_fec.data", "bn_fec_short_unwrap.dcd"]

from pkg_resources import resource_filename

bn_fec_data = resource_filename(__name__, "data/bn_fec.data")
bn_fec_dcd_unwrap = resource_filename(__name__, "data/bn_fec_short_unwrap.dcd")
bn_fec_dcd_wrap = resource_filename(__name__, "data/bn_fec_short_wrap.dcd")

del resource_filename
