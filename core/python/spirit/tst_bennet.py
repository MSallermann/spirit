import spirit.spiritlib as spiritlib
import spirit.parameters as parameters
import spirit.system as system
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

_Calculate          = _spirit.TST_Bennet_Calculate
_Calculate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,  ctypes.c_int, ctypes.c_int]
_Calculate.restype  = ctypes.c_float
def calculate(p_state, idx_image_minimum, idx_image_sp, n_iterations_bennet=5000, idx_chain=-1):
    """Performs an HTST calculation and returns rate prefactor.

    *Note:* this function must be called before any of the getters.
    """
    return _Calculate(p_state, idx_image_minimum, idx_image_sp, n_iterations_bennet, idx_chain)

_Get_Info = _spirit.TST_Bennet_Get_Info = _spirit.TST_Bennet_Get_Info
_Get_Info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Info.restype  = None

def get_info_dict(p_state, idx_chain = -1):
    benn_min = ctypes.c_float()
    err_benn_min = ctypes.c_float()

    benn_sp = ctypes.c_float()
    err_benn_sp = ctypes.c_float()

    unstable_mode_contribution = ctypes.c_float()

    rate = ctypes.c_float()
    err_rate = ctypes.c_float()

    _Get_Info(ctypes.c_void_p(p_state), ctypes.byref(benn_min), ctypes.byref(err_benn_min), ctypes.byref(benn_sp), ctypes.byref(err_benn_sp), ctypes.byref(unstable_mode_contribution), ctypes.byref(rate), ctypes.byref(err_rate), ctypes.c_int(idx_chain))

    return {
        "benn_min" : benn_min.value,
        "err_benn_min" : err_benn_min.value,
        "benn_sp" : benn_sp.value,
        "err_benn_sp" : err_benn_sp.value,
        "unstable_mode_contribution" : unstable_mode_contribution.value,
        "rate" : rate.value,
        "err_rate" : err_rate.value
    }