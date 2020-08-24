#include <Spirit/System.h>
#include <Spirit/State.h>
#include <data/State.hpp>
#include <engine/Eigenmodes.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

int System_Get_Index(State * state) noexcept
try
{
    return state->idx_active_image;
}
catch( ... )
{
    spirit_handle_exception_api(-1, -1);
    return -1;
}

int System_Get_NOS(State * state, int idx_image, int idx_chain) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    return image->nos;
}
catch( ... )
{
    spirit_handle_exception_api(idx_image, idx_chain);
    return 0;
}

scalar * System_Get_Spin_Directions(State * state, int idx_image, int idx_chain) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    return (scalar *)(*image->spins)[0].data();
}
catch( ... )
{
    spirit_handle_exception_api(idx_image, idx_chain);
    return nullptr;
}

scalar * System_Get_Effective_Field(State * state, int idx_image, int idx_chain) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    //printf("%f\n", image->effective_field[500*500*3*2]);
    return  &image->hamiltonian->gradient_contributions_per_spin[image->geometry->nos * 3 * 0];
}
catch( ... )
{
    spirit_handle_exception_api(idx_image, idx_chain);
    return nullptr;
}
#ifndef SPIRIT_LOW_MEMORY
	scalar * System_Get_Exchange_Field(State * state, int idx_image, int idx_chain) noexcept
	try
	{
	    std::shared_ptr<Data::Spin_System> image;
	    std::shared_ptr<Data::Spin_System_Chain> chain;

	    // Fetch correct indices and pointers
	    from_indices( state, idx_image, idx_chain, image, chain );
	    return  &image->hamiltonian->gradient_contributions_per_spin[image->geometry->nos * 3 * 4];
	}
	catch( ... )
	{
	    spirit_handle_exception_api(idx_image, idx_chain);
	    return nullptr;
	}

	scalar * System_Get_Anisotropy_Field(State * state, int idx_image, int idx_chain) noexcept
	try
	{
	    std::shared_ptr<Data::Spin_System> image;
	    std::shared_ptr<Data::Spin_System_Chain> chain;

	    // Fetch correct indices and pointers
	    from_indices( state, idx_image, idx_chain, image, chain );
	    return  &image->hamiltonian->gradient_contributions_per_spin[image->geometry->nos * 3 * 3];
	}
	catch( ... )
	{
	    spirit_handle_exception_api(idx_image, idx_chain);
	    return nullptr;
	}

	scalar * System_Get_DMI_Field(State * state, int idx_image, int idx_chain) noexcept
	try
	{
	    std::shared_ptr<Data::Spin_System> image;
	    std::shared_ptr<Data::Spin_System_Chain> chain;

	    // Fetch correct indices and pointers
	    from_indices( state, idx_image, idx_chain, image, chain );
	    return  &image->hamiltonian->gradient_contributions_per_spin[image->geometry->nos * 3 * 5];
	}
	catch( ... )
	{
	    spirit_handle_exception_api(idx_image, idx_chain);
	    return nullptr;
	}

	scalar * System_Get_DDI_Field(State * state, int idx_image, int idx_chain) noexcept
	try
	{
	    std::shared_ptr<Data::Spin_System> image;
	    std::shared_ptr<Data::Spin_System_Chain> chain;

	    // Fetch correct indices and pointers
	    from_indices( state, idx_image, idx_chain, image, chain );
	    return  &image->hamiltonian->gradient_contributions_per_spin[image->geometry->nos * 3 * 6];
	}
	catch( ... )
	{
	    spirit_handle_exception_api(idx_image, idx_chain);
	    return nullptr;
	}

	scalar * System_Get_Zeeman_Field(State * state, int idx_image, int idx_chain) noexcept
	try
	{
	    std::shared_ptr<Data::Spin_System> image;
	    std::shared_ptr<Data::Spin_System_Chain> chain;

	    // Fetch correct indices and pointers
	    from_indices( state, idx_image, idx_chain, image, chain );
	    return &image->hamiltonian->gradient_contributions_per_spin[image->geometry->nos * 3 * 2];
	}
	catch( ... )
	{
	    spirit_handle_exception_api(idx_image, idx_chain);
	    return nullptr;
	}

    scalar* System_Get_Torque(State* state, int idx_image, int idx_chain) noexcept
        try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices(state, idx_image, idx_chain, image, chain);
        return &image->hamiltonian->gradient_contributions_per_spin[image->geometry->nos * 3 * 1];
    }
    catch (...)
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return nullptr;
    }
#endif
scalar * System_Get_Eigenmode(State * state, int idx_mode, int idx_image, int idx_chain) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // Check mode index
    if (idx_mode >= image->modes.size())
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
            fmt::format("Invalid mode index {}, image has only {} modes stored.", idx_mode, image->modes.size()) );
        return nullptr;
    }

    // Check if mode has been calculated
    if (!image->modes[idx_mode])
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
            fmt::format("Mode {} has not yet been calculated.", idx_mode) );
        return nullptr;
    }

    return (*image->modes[idx_mode])[0].data();
}
catch( ... )
{
    spirit_handle_exception_api(idx_image, idx_chain);
    return nullptr;
}

float System_Get_Rx(State * state, int idx_image, int idx_chain) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    return (float)chain->Rx[idx_image];
}
catch( ... )
{
    spirit_handle_exception_api(idx_image, idx_chain);
    return 0;
}

float System_Get_Energy(State * state, int idx_image, int idx_chain) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    return (float)image->E;
}
catch( ... )
{
    spirit_handle_exception_api(idx_image, idx_chain);
    return 0;
}

void System_Get_Energy_Array(State * state, float * energies, int idx_image, int idx_chain) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    for (unsigned int i=0; i<image->E_array.size(); ++i)
    {
        energies[i] = (float)image->E_array[i].second;
    }
}
catch( ... )
{
    spirit_handle_exception_api(idx_image, idx_chain);
}

void System_Get_Eigenvalues(State * state, float * eigenvalues, int idx_image, int idx_chain) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    for (unsigned int i=0; i<image->eigenvalues.size(); ++i)
    {
        eigenvalues[i] = (float)image->eigenvalues[i];
    }
}
catch( ... )
{
    spirit_handle_exception_api(idx_image, idx_chain);
}

void System_Print_Energy_Array(State * state, int idx_image, int idx_chain) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    scalar nd = 1/(scalar)image->nos;

    std::cerr << "E_tot = " << image->E*nd << "  ||  ";

    for (unsigned int i=0; i<image->E_array.size(); ++i)
    {
        std::cerr << image->E_array[i].first << " = " << image->E_array[i].second*nd;
        if (i < image->E_array.size()-1) std::cerr << "  |  ";
    }
    std::cerr << std::endl;
}
catch( ... )
{
    spirit_handle_exception_api(idx_image, idx_chain);
}

void System_Update_Data(State * state, int idx_image, int idx_chain) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    try
    {
        image->UpdateEnergy();
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api(idx_image, idx_chain);
}

// Calculate the eigenmodes of the System
void System_Update_Eigenmodes(State *state, int idx_image, int idx_chain) noexcept
try
{
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    Engine::Eigenmodes::Calculate_Eigenmodes(image, idx_image, idx_chain );
    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api(idx_image, idx_chain);
}
