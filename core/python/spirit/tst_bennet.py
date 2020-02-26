import spirit.spiritlib as spiritlib
import spirit.parameters as parameters
import spirit.system as system
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

_Calculate          = _spirit.TST_Bennet_Calculate
_Calculate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Calculate.restype  = ctypes.c_float
def calculate(p_state, idx_image_minimum, idx_image_sp, n_chain=2, n_iterations_bennet=5000, idx_chain=-1):
    """Performs an HTST calculation and returns rate prefactor.

    *Note:* this function must be called before any of the getters.
    """
    return _Calculate(p_state, idx_image_minimum, idx_image_sp, n_chain, n_iterations_bennet, idx_chain)

_Get_Info = _spirit.TST_Bennet_Get_Info = _spirit.TST_Bennet_Get_Info
_Get_Info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Info.restype  = None

def get_info_dict(p_state, idx_chain = -1):
    Z_ratio = ctypes.c_float()
    err_Z_ratio = ctypes.c_float()

    vel_perp = ctypes.c_float()
    err_vel_perp = ctypes.c_float()

    unstable_mode_contribution = ctypes.c_float()

    rate = ctypes.c_float()
    err_rate = ctypes.c_float()

    _Get_Info(ctypes.c_void_p(p_state), ctypes.byref(Z_ratio), ctypes.byref(err_Z_ratio), ctypes.byref(vel_perp), ctypes.byref(err_vel_perp), ctypes.byref(unstable_mode_contribution), ctypes.byref(rate), ctypes.byref(err_rate), ctypes.c_int(idx_chain))

    return {
        "Z_ratio" : Z_ratio.value,
        "err_Z_ratio" : err_Z_ratio.value,
        "vel_perp" : vel_perp.value,
        "err_vel_perp" : err_vel_perp.value,
        "unstable_mode_contribution" : unstable_mode_contribution.value,
        "rate" : rate.value,
        "err_rate" : err_rate.value
    }