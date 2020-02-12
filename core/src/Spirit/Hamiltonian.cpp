#include <Spirit/Hamiltonian.h>

#include <data/State.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Hamiltonian_Heisenberg.hpp>
#include <engine/Hamiltonian_Micromagnetic.hpp>
#include <engine/Hamiltonian_Gaussian.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <fmt/format.h>

using namespace Utility;

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

void Hamiltonian_Set_Kind(State *state, Hamiltonian_Type type, int idx_chain) noexcept
try
{
    // TODO
    if( type != Hamiltonian_Heisenberg && type != Hamiltonian_Micromagnetic && type != Hamiltonian_Gaussian )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API, fmt::format(
            "Hamiltonian_Set_Kind: unknown type index {}", int(type)), -1, idx_chain );
        return;
    }

    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    int idx_image = -1;
    from_indices( state, idx_image, idx_chain, image, chain );

    idx_image = 0;
    std::string kind_str = "";
    if( type == Hamiltonian_Heisenberg )
    {
        kind_str = "Heisenberg";

        if( kind_str == image->hamiltonian->Name() )
        {
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, fmt::format(
                "Hamiltonian is already of {} kind. Not doing anything.", kind_str), -1, idx_chain );
            return;
        }
    }
    else if( type == Hamiltonian_Micromagnetic )
    {
        kind_str = "Micromagnetic";

        if( kind_str == image->hamiltonian->Name() )
        {
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, fmt::format(
                "Hamiltonian is already of {} kind. Not doing anything.", kind_str), -1, idx_chain );
            return;
        }
    }
    else if( type == Hamiltonian_Gaussian )
    {
        kind_str = "Gaussian";

        if( kind_str == image->hamiltonian->Name() )
        {
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, fmt::format(
                "Hamiltonian is already of {} kind. Not doing anything.", kind_str), -1, idx_chain );
            return;
        }

        Log( Utility::Log_Level::Error, Utility::Log_Sender::API, fmt::format(
            "Cannot yet set Hamiltonian kind to {} - this function is only a stub!", kind_str), -1, idx_chain );
        return;
    }

    for( auto& image : chain->images )
    {
        image->Lock();
        try
        {
            if( type == Hamiltonian_Heisenberg )
            {
                // TODO: are these the desired defaults?
                image->hamiltonian = std::shared_ptr<Engine::Hamiltonian>(new Engine::Hamiltonian_Heisenberg(
                    0, Vector3{0, 0, 1},
                    {}, {}, {},
                    {}, {}, SPIRIT_CHIRALITY_NEEL, Engine::DDI_Method::None,
                    {0, 0, 0}, true, 0, {}, {},
                    image->geometry,
                    image->hamiltonian->boundary_conditions));
            }
            else if( type == Hamiltonian_Micromagnetic )
            {
                // TODO: are these the desired defaults?
                /*
                image->hamiltonian = std::shared_ptr<Engine::Hamiltonian>(new Engine::Hamiltonian_Micromagnetic(
                    0, Vector3{0, 0, 1},
                    1, scalarfield(1,0),vectorfield(1, Vector3{0,0,1}),
                    0,
                    0,
                    image->geometry,
                    2,
                    image->hamiltonian->boundary_conditions,
                    Vector3{5e-11, 5e-11, 5e-10},
                    8e5
                    ));*/
            }
            else if( type == Hamiltonian_Gaussian )
            {
                // TODO
                // image->hamiltonian = std::shared_ptr<...>(new Engine::Hamiltonian_Gaussian(...));
            }
        }
        catch( ... )
        {
            spirit_handle_exception_api(idx_image, idx_chain);
        }
        image->Unlock();
        ++idx_image;
    }

    Log( Utility::Log_Level::All, Utility::Log_Sender::API, fmt::format(
        "Set Hamiltonian kind to {}", kind_str), -1, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api(-1, idx_chain);
}

void Hamiltonian_Set_Boundary_Conditions(State *state, const bool * periodical, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        image->Lock();
        try
        {
            image->hamiltonian->boundary_conditions[0] = periodical[0];
            image->hamiltonian->boundary_conditions[1] = periodical[1];
            image->hamiltonian->boundary_conditions[2] = periodical[2];
        }
        catch( ... )
        {
            spirit_handle_exception_api(idx_image, idx_chain);
        }
        image->Unlock();

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
            fmt::format("Set boundary conditions to {} {} {}", periodical[0], periodical[1], periodical[2]),
            idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Hamiltonian_Set_Cell_Sizes(State *state, const float * cell_sizes, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        // Lock mutex because simulations may be running
        image->Lock();

        try
        {
            auto ham = (Engine::Hamiltonian_Micromagnetic*)image->hamiltonian.get();

            // Into the Hamiltonian
            ham->cell_sizes[0] = cell_sizes[0];
            ham->cell_sizes[1] = cell_sizes[1];
            ham->cell_sizes[2] = cell_sizes[2];
            // Update Energies
            ham->Update_Energy_Contributions();

        }
        catch( ... )
        {
            spirit_handle_exception_api(idx_image, idx_chain);
        }

        // Unlock mutex
        image->Unlock();

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
            fmt::format("Set cell sizes to ({}, {}, {})", cell_sizes[0], cell_sizes[1], cell_sizes[2]),
            idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Hamiltonian_Set_Ms(State *state, const float Ms, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        // Lock mutex because simulations may be running
        image->Lock();

        try
        {

            auto ham = (Engine::Hamiltonian_Micromagnetic*)image->hamiltonian.get();

            // Into the Hamiltonian
            ham->Ms = Ms;
            // Update Energies
            ham->Update_Energy_Contributions();

        }
        catch( ... )
        {
            spirit_handle_exception_api(idx_image, idx_chain);
        }

        // Unlock mutex
        image->Unlock();

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
            fmt::format("Set Ms to {}", Ms),
            idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Hamiltonian_Set_Field(State *state, float magnitude, const float * normal, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        if( image->hamiltonian->Name() != "Heisenberg" && image->hamiltonian->Name() != "Micromagnetic" )
        {
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                "External field cannot be set on " + image->hamiltonian->Name(), idx_image, idx_chain );
            return;
        }

        // Lock mutex because simulations may be running
        image->Lock();

        try
        {
            // Set
            if (image->hamiltonian->Name() == "Heisenberg")
            {
                auto ham = (Engine::Hamiltonian_Heisenberg*)image->hamiltonian.get();

                // Normals
                Vector3 new_normal{normal[0], normal[1], normal[2]};
                new_normal.normalize();

                // Into the Hamiltonian
                ham->external_field_magnitude = magnitude * Constants::mu_B;
                ham->external_field_normal = new_normal;

                // Update Energies
                ham->Update_Energy_Contributions();
            }

        }
        catch( ... )
        {
            spirit_handle_exception_api(idx_image, idx_chain);
        }

        // Unlock mutex
        image->Unlock();

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
            fmt::format("Set external field to {}, direction ({}, {}, {})", magnitude, normal[0], normal[1], normal[2]),
            idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Hamiltonian_Set_Anisotropy( State *state, float magnitude, const float * normal,
                                 int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        image->Lock();

        try
        {
            if (image->hamiltonian->Name() == "Heisenberg")
            {
                auto ham = (Engine::Hamiltonian_Heisenberg*)image->hamiltonian.get();
                int nos = image->nos;
                int n_cell_atoms = image->geometry->n_cell_atoms;

                // Indices and Magnitudes
                intfield new_indices(n_cell_atoms);
                scalarfield new_magnitudes(n_cell_atoms);
                for (int i = 0; i<n_cell_atoms; ++i)
                {
                    new_indices[i] = i;
                    new_magnitudes[i] = magnitude;
                }
                // Normals
                Vector3 new_normal{ normal[0], normal[1], normal[2] };
                new_normal.normalize();
                vectorfield new_normals(nos, new_normal);

                // Into the Hamiltonian
                ham->anisotropy_indices = new_indices;
                ham->anisotropy_magnitudes = new_magnitudes;
                ham->anisotropy_normals = new_normals;

                // Update Energies
                ham->Update_Energy_Contributions();

                Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                    fmt::format("Set anisotropy to {}, direction ({}, {}, {})", magnitude, normal[0], normal[1], normal[2]),
                    idx_image, idx_chain );
            } else {
                Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                        "Anisotropy cannot be set on " + image->hamiltonian->Name(), idx_image, idx_chain);
            }
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
}



void Hamiltonian_Set_Exchange(State *state, int n_shells, const float* jij, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        image->Lock();

        try
        {
            if (image->hamiltonian->Name() == "Heisenberg")
            {
                // Update the Hamiltonian
                auto ham = (Engine::Hamiltonian_Heisenberg*)image->hamiltonian.get();
                ham->exchange_shell_magnitudes = scalarfield(jij, jij + n_shells);
                ham->exchange_pairs_in         = pairfield(0);
                ham->exchange_magnitudes_in    = scalarfield(0);
                ham->Update_Interactions();

                std::string message = fmt::format("Set exchange to {} shells", n_shells);
                if (n_shells > 0) message += fmt::format(" Jij[0] = {}", jij[0]);
                Log(Utility::Log_Level::Info, Utility::Log_Sender::API, message, idx_image, idx_chain);
            }
            else
                Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                        "Exchange cannot be set on " + image->hamiltonian->Name(), idx_image, idx_chain );
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
}

void Hamiltonian_Set_Exchange_Stiffness(State *state, float exchange_stiffness, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        image->Lock();

        try
        {
            if (image->hamiltonian->Name() == "Micromagnetic")
            {
                // Update the Hamiltonian
                auto ham = (Engine::Hamiltonian_Micromagnetic*)image->hamiltonian.get();
                ham->exchange_stiffness = exchange_stiffness;

                ham->Update_Interactions();

                std::string message = fmt::format("Set exchange stiffness:\n");
                message += fmt::format("{}\n", exchange_stiffness);
                Log(Utility::Log_Level::Info, Utility::Log_Sender::API, message, idx_image, idx_chain);
            }
            else
                Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                        "Exchange cannot be set on " + image->hamiltonian->Name(), idx_image, idx_chain );
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
}

void Hamiltonian_Set_DMI(State *state, int n_shells, const float * dij, int chirality, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        image->Lock();

        if( chirality != SPIRIT_CHIRALITY_BLOCH         &&
            chirality != SPIRIT_CHIRALITY_NEEL          &&
            chirality != SPIRIT_CHIRALITY_BLOCH_INVERSE &&
            chirality != SPIRIT_CHIRALITY_NEEL_INVERSE  )
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::API, fmt::format(
                "Hamiltonian_Set_DMI: Invalid DM chirality {}", chirality), idx_image, idx_chain );
            return;
        }

        try
        {
            if (image->hamiltonian->Name() == "Heisenberg")
            {
                // Update the Hamiltonian
                auto ham = (Engine::Hamiltonian_Heisenberg*)image->hamiltonian.get();
                ham->dmi_shell_magnitudes = scalarfield(dij, dij + n_shells);
                ham->dmi_shell_chirality  = chirality;
                ham->dmi_pairs_in         = pairfield(0);
                ham->dmi_magnitudes_in    = scalarfield(0);
                ham->dmi_normals_in       = vectorfield(0);
                ham->Update_Interactions();

                std::string message = fmt::format("Set dmi to {} shells", n_shells);
                if (n_shells > 0) message += fmt::format(" Dij[0] = {}", dij[0]);
                Log(Utility::Log_Level::Info, Utility::Log_Sender::API, message, idx_image, idx_chain);
            }
            else
                Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, "DMI cannot be set on " +
                        image->hamiltonian->Name(), idx_image, idx_chain );
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
}

void Hamiltonian_Set_Spiralisation_Constant(State *state, float dmi_constant, int dmi_chirality, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        image->Lock();

        try
        {
            if (image->hamiltonian->Name() == "Micromagnetic")
            {
                // Update the Hamiltonian
                auto ham = (Engine::Hamiltonian_Micromagnetic*)image->hamiltonian.get();
                ham -> dmi_constant = dmi_constant;
                ham -> dmi_chirality = dmi_chirality;
                ham->Update_Interactions();

                std::string message = fmt::format("Set spiralisation constant:\n");
                message += fmt::format("{}\n", dmi_constant);
                Log(Utility::Log_Level::Info, Utility::Log_Sender::API, message, idx_image, idx_chain);
            }
            else
                Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                        "Spiralisation constant cannot be set on " + image->hamiltonian->Name(), idx_image, idx_chain );
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
}

void Hamiltonian_Set_DDI(State *state, int ddi_method, int n_periodic_images[3], float cutoff_radius, bool pb_zero_padding, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        image->Lock();

        try
        {
            if (image->hamiltonian->Name() == "Heisenberg")
            {
                auto ham = (Engine::Hamiltonian_Heisenberg*)image->hamiltonian.get();

                ham->ddi_method = Engine::DDI_Method(ddi_method);
                ham->ddi_n_periodic_images[0] = n_periodic_images[0];
                ham->ddi_n_periodic_images[1] = n_periodic_images[1];
                ham->ddi_n_periodic_images[2] = n_periodic_images[2];
                ham->ddi_cutoff_radius = cutoff_radius;
                ham->ddi_pb_zero_padding = pb_zero_padding;
                ham->Update_Interactions();

                Log( Utility::Log_Level::Info, Utility::Log_Sender::API, fmt::format(
                    "Set ddi to method {}, periodic images {} {} {}, cutoff radius {} and pb_zero_padding {}",
                    ddi_method, n_periodic_images[0], n_periodic_images[1], n_periodic_images[2], cutoff_radius, pb_zero_padding), idx_image, idx_chain );
            }
            else
                Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, "DDI cannot be set on " + 
                        image->hamiltonian->Name(), idx_image, idx_chain );
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
}

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

const char * Hamiltonian_Get_Name(State * state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        return image->hamiltonian->Name().c_str();
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return nullptr;
    }
}

void Hamiltonian_Get_Boundary_Conditions(State *state, bool * periodical, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        periodical[0] = image->hamiltonian->boundary_conditions[0];
        periodical[1] = image->hamiltonian->boundary_conditions[1];
        periodical[2] = image->hamiltonian->boundary_conditions[2];
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Hamiltonian_Get_Ms(State *state, float * Ms, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        auto ham = (Engine::Hamiltonian_Micromagnetic*)image->hamiltonian.get();
        // Magnitude
        *Ms = float(ham->Ms);
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Hamiltonian_Get_Cell_Sizes(State *state, float * cell_sizes, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        auto ham = (Engine::Hamiltonian_Micromagnetic*)image->hamiltonian.get();
        // Magnitude
        cell_sizes[0] = (float)(ham->cell_sizes[0]);
        cell_sizes[1] = (float)(ham->cell_sizes[1]);
        cell_sizes[2] = (float)(ham->cell_sizes[2]);
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Hamiltonian_Get_Field(State *state, float * magnitude, float * normal, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        if (image->hamiltonian->Name() == "Heisenberg")
        {
            auto ham = (Engine::Hamiltonian_Heisenberg*)image->hamiltonian.get();

            if (ham->external_field_magnitude > 0)
            {
                // Magnitude
                *magnitude = (float)(ham->external_field_magnitude / Constants::mu_B);

                // Normal
                normal[0] = (float)ham->external_field_normal[0];
                normal[1] = (float)ham->external_field_normal[1];
                normal[2] = (float)ham->external_field_normal[2];
            }
            else
            {
                *magnitude = 0;
                normal[0] = 0;
                normal[1] = 0;
                normal[2] = 1;
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Hamiltonian_Get_Anisotropy(State *state, float * magnitude, float * normal, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        if (image->hamiltonian->Name() == "Heisenberg")
        {
            auto ham = (Engine::Hamiltonian_Heisenberg*)image->hamiltonian.get();

            if (ham->anisotropy_indices.size() > 0)
            {
                // Magnitude
                *magnitude = (float)ham->anisotropy_magnitudes[0];

                // Normal
                normal[0] = (float)ham->anisotropy_normals[0][0];
                normal[1] = (float)ham->anisotropy_normals[0][1];
                normal[2] = (float)ham->anisotropy_normals[0][2];
            }
            else
            {
                *magnitude = 0;
                normal[0] = 0;
                normal[1] = 0;
                normal[2] = 1;
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Hamiltonian_Get_Exchange_Shells(State *state, int * n_shells, float * jij, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        if (image->hamiltonian->Name() == "Heisenberg")
        {
            auto ham = (Engine::Hamiltonian_Heisenberg*)image->hamiltonian.get();

            *n_shells = ham->exchange_shell_magnitudes.size();

            // Note the array needs to be correctly allocated beforehand!
            for (int i=0; i<ham->exchange_shell_magnitudes.size(); ++i)
            {
                jij[i] = (float)ham->exchange_shell_magnitudes[i];
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

int  Hamiltonian_Get_Exchange_N_Pairs(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
            image->hamiltonian->Name() + " Hamiltonian: fetching exchange pairs is not yet implemented...", idx_image, idx_chain );
        return 0;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}

void Hamiltonian_Get_Exchange_Pairs(State *state, float * idx[2], float * translations[3], float * Jij, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
            image->hamiltonian->Name() + " Hamiltonian: fetching exchange pairs is not yet implemented...", idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Hamiltonian_Get_Exchange_Stiffness(State *state, float * exchange_stiffness, int region_id, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        if (image->hamiltonian->Name() == "Micromagnetic")
        {
            auto ham = (Engine::Hamiltonian_Micromagnetic*)image->hamiltonian.get();
            *exchange_stiffness=ham->exchange_stiffness;
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Hamiltonian_Get_DMI_Shells(State *state, int * n_shells, float * dij, int * chirality, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        if (image->hamiltonian->Name() == "Heisenberg")
        {
            auto ham = (Engine::Hamiltonian_Heisenberg*)image->hamiltonian.get();

            *n_shells  = ham->dmi_shell_magnitudes.size();
            *chirality = ham->dmi_shell_chirality;

            for (int i=0; i<*n_shells; ++i)
            {
                dij[i] = (float)ham->dmi_shell_magnitudes[i];
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

int  Hamiltonian_Get_DMI_N_Pairs(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
            image->hamiltonian->Name() + " Hamiltonian: fetching DMI pairs is not yet implemented...", idx_image, idx_chain );
        return 0;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}

void Hamiltonian_Get_Spiralisation_Constant(State *state, float * dmi_constant, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        if (image->hamiltonian->Name() == "Micromagnetic")
        {
            auto ham = (Engine::Hamiltonian_Micromagnetic*)image->hamiltonian.get();

            *dmi_constant = ham->dmi_constant;
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Hamiltonian_Get_DDI(State *state, int * ddi_method, int n_periodic_images[3], float * cutoff_radius, bool * pb_zero_padding, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        if (image->hamiltonian->Name() == "Heisenberg")
        {
            auto ham = (Engine::Hamiltonian_Heisenberg*)image->hamiltonian.get();

            *ddi_method          = (int)ham->ddi_method;
            n_periodic_images[0] = (int)ham->ddi_n_periodic_images[0];
            n_periodic_images[1] = (int)ham->ddi_n_periodic_images[1];
            n_periodic_images[2] = (int)ham->ddi_n_periodic_images[2];
            *cutoff_radius       = (float)ham->ddi_cutoff_radius;
            *pb_zero_padding     = ham->ddi_pb_zero_padding;
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}