#include <Spirit/TST_Bennet.h>
#include <engine/TST_Bennet.hpp>
#include <data/State.hpp>
#include <engine/HTST.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

float TST_Bennet_Calculate(State * state, int idx_image_minimum, int idx_image_sp, int idx_chain) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image_minimum, image_sp;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image_minimum, idx_chain, image_minimum, chain);
    from_indices(state, idx_image_sp, idx_chain, image_sp, chain);

    auto& info = chain->htst_info;
    info.minimum = image_minimum;
    info.saddle_point = image_sp;

    Engine::TST_Bennet::Calculate(chain->htst_info);

    return (float)info.prefactor;
}
catch( ... )
{
    spirit_handle_exception_api(-1, idx_chain);
    return 0;
}