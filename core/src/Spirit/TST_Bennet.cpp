#include <Spirit/TST_Bennet.h>
#include <engine/TST_Bennet.hpp>
#include <data/State.hpp>
#include <engine/HTST.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

float TST_Bennet_Calculate(State * state, int idx_image_minimum, int idx_image_sp, int n_chain, int n_iterations_bennet, int idx_chain) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image_minimum, image_sp;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image_minimum, idx_chain, image_minimum, chain);
    from_indices(state, idx_image_sp, idx_chain, image_sp, chain);

    auto& info = chain->tst_bennet_info;
    info.minimum = image_minimum;
    info.saddle_point = image_sp;

    Engine::TST_Bennet::Calculate(chain->tst_bennet_info, n_chain, n_iterations_bennet);

    return (float)0;
}
catch( ... )
{
    spirit_handle_exception_api(-1, idx_chain);
    return 0;
}

void TST_Bennet_Get_Info(State * state, float * Z_ratio, float * err_Z_ratio, float * vel_perp, float * err_vel_perp, float * unstable_mode_contribution, float * rate, float * rate_err, int idx_chain) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    auto& info = chain->tst_bennet_info;

    *Z_ratio = info.Z_ratio;
    *err_Z_ratio = info.err_Z_ratio;
    *vel_perp = info.vel_perp;
    *err_vel_perp = info.vel_perp_err;
    *unstable_mode_contribution = info.unstable_mode_contribution;
    *rate = info.rate;
    *rate_err = info.rate_err;
}
catch( ... )
{
    spirit_handle_exception_api(-1, idx_chain);
}