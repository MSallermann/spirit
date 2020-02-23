#include <Spirit/TST_Bennet.h>
#include <engine/TST_Bennet.hpp>
#include <data/State.hpp>
#include <engine/HTST.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

float TST_Bennet_Calculate(State * state, int idx_image_minimum, int idx_image_sp, int n_iterations_bennet, int idx_chain) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image_minimum, image_sp;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image_minimum, idx_chain, image_minimum, chain);
    from_indices(state, idx_image_sp, idx_chain, image_sp, chain);

    auto& info = chain->tst_bennet_info;
    info.minimum = image_minimum;
    info.saddle_point = image_sp;

    Engine::TST_Bennet::Calculate(chain->tst_bennet_info, n_iterations_bennet);

    return (float)0;
}
catch( ... )
{
    spirit_handle_exception_api(-1, idx_chain);
    return 0;
}

void TST_Bennet_Get_Info(State * state, float * benn_min, float * err_benn_min, float * benn_sp, float * err_benn_sp, float * unstable_mode_contribution, float * rate, float * rate_err, int idx_chain) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    auto& info = chain->tst_bennet_info;

    *benn_min = info.benn_min;
    *err_benn_min = info.err_benn_min;
    *benn_sp = info.benn_sp;
    *err_benn_sp = info.err_benn_sp;
    *unstable_mode_contribution = info.unstable_mode_contribution;
    *rate = info.rate;
    *rate_err = info.rate_err;
}
catch( ... )
{
    spirit_handle_exception_api(-1, idx_chain);
}