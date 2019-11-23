import spirit.spiritlib as spiritlib
import spirit.parameters as parameters
import spirit.system as system
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

_Calculate          = _spirit.TST_Bennet_Calculate
_Calculate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Calculate.restype  = ctypes.c_float
def calculate(p_state, idx_image_minimum, idx_image_sp, idx_chain=-1):
    """Performs an HTST calculation and returns rate prefactor.

    *Note:* this function must be called before any of the getters.
    """
    return _Calculate(p_state, idx_image_minimum, idx_image_sp, idx_chain)