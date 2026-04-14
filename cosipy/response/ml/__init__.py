import cosipy

if not cosipy.with_ml:
    raise ImportError("Install cosipy with [ml] optional packages to use these features.")

from .NFResponse import NFResponse
from .nf_instrument_response_function import UnpolarizedNFFarFieldInstrumentResponseFunction