﻿#include <io/IO.hpp>
#include <io/Filter_File_Handle.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <ctime>

#include <fmt/format.h>
#include <fmt/ostream.h>

using namespace Utility;

namespace IO
{
    void Log_from_Config(const std::string configFile, bool force_quiet)
    {
        // Verbosity and Reject Level are read as integers
        int i_level_file = 5, i_level_console = 5;
        std::string output_folder = ".";
        std::string file_tag = "";
        bool messages_to_file    = true,
             messages_to_console = true,
             save_input_initial  = false,
             save_input_final    = false,
             save_positions_initial  = false,
             save_positions_final    = false,
             save_neighbours_initial = false,
             save_neighbours_final   = false;

        // "Quiet" settings
        if (force_quiet)
        {
            // Don't save the Log to file
            Log.messages_to_file = false;
            // Don't print the Log to console
            Log.messages_to_console = false;
            // Don't save input configs
            Log.save_input_initial = false;
            Log.save_input_final = false;
            // Don't save positions
            Log.save_positions_initial = false;
            Log.save_positions_final = false;
            // Don't save neighbours
            Log.save_neighbours_initial = false;
            Log.save_neighbours_final = false;
            // Don't print messages, except Error & Severe
            Log.level_file = Utility::Log_Level::Error;
            Log.level_console = Utility::Log_Level::Error;
        }

        //------------------------------- Parser --------------------------------
        if( configFile != "" )
        {
            try
            {
                Log(Log_Level::Info, Log_Sender::IO, "Building Log");
                IO::Filter_File_Handle myfile(configFile);

                // Time tag
                myfile.Read_Single(file_tag, "output_file_tag");

                // Output folder
                myfile.Read_Single(output_folder, "log_output_folder");

                // Save Output (Log Messages) to file
                myfile.Read_Single(messages_to_file, "log_to_file");
                // File Accept Level
                myfile.Read_Single(i_level_file, "log_file_level");

                // Print Output (Log Messages) to console
                myfile.Read_Single(messages_to_console, "log_to_console");
                // File Accept Level
                myfile.Read_Single(i_level_console, "log_console_level");

                // Save Input (parameters from config file and defaults) on State Setup
                myfile.Read_Single(save_input_initial, "save_input_initial");
                // Save Input (parameters from config file and defaults) on State Delete
                myfile.Read_Single(save_input_final, "save_input_final");

                // Save Input (parameters from config file and defaults) on State Setup
                myfile.Read_Single(save_positions_initial, "save_positions_initial");
                // Save Input (parameters from config file and defaults) on State Delete
                myfile.Read_Single(save_positions_final, "save_positions_final");

                 // Save Input (parameters from config file and defaults) on State Setup
                 myfile.Read_Single(save_neighbours_initial, "save_neighbours_initial");
                 // Save Input (parameters from config file and defaults) on State Delete
                 myfile.Read_Single(save_neighbours_final, "save_neighbours_final");

            }// end try
            catch( ... )
            {
                spirit_rethrow(	fmt::format("Failed to read Log Levels from file \"{}\". Leaving values at default.", configFile) );
            }
        }

        // Log the parameters
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("File tag on output     = {}", file_tag));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log output folder      = \"{}\"", output_folder));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log to file            = {}", messages_to_file));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log file accept level  = {}", i_level_file));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log to console         = {}", messages_to_console));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log print accept level = {}", i_level_console));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log input save initial = {}", save_input_initial));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log input save final   = {}", save_input_final));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log positions save initial  = {}", save_positions_initial));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log positions save final    = {}", save_positions_final));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log neighbours save initial = {}", save_neighbours_initial));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log neighbours save final   = {}", save_neighbours_final));

        // Update the Log
        if (!force_quiet)
        {
            Log.level_file    = Log_Level(i_level_file);
            Log.level_console = Log_Level(i_level_console);

            Log.messages_to_file    = messages_to_file;
            Log.messages_to_console = messages_to_console;
            Log.save_input_initial  = save_input_initial;
            Log.save_input_final    = save_input_final;
            Log.save_positions_initial  = save_positions_initial;
            Log.save_positions_final    = save_positions_final;
            Log.save_neighbours_initial = save_neighbours_initial;
            Log.save_neighbours_final   = save_neighbours_final;
        }

        Log.file_tag      = file_tag;
        Log.output_folder = output_folder;

        if ( file_tag == "<time>" )
            Log.fileName = "Log_" + Utility::Timing::CurrentDateTime() + ".txt";
        else if ( file_tag != "" )
            Log.fileName = "Log_" + file_tag + ".txt";
        else
            Log.fileName = "Log.txt";

    }// End Log_Levels_from_Config


    std::unique_ptr<Data::Spin_System> Spin_System_from_Config(std::string configFile)
    {
        // Parse
        try
        {
            Log(Log_Level::Info, Log_Sender::IO, "-------------- Initialising Spin System ------------");

            // Geometry
            auto geometry = Geometry_from_Config(configFile);
            // LLG Parameters
            auto llg_params = Parameters_Method_LLG_from_Config(configFile);
            // MC Parameters
            auto mc_params = Parameters_Method_MC_from_Config(configFile);
            // EMA Parameters
            auto ema_params = Parameters_Method_EMA_from_Config(configFile);
            // MMF Parameters
            auto mmf_params = Parameters_Method_MMF_from_Config(configFile);
            // Hamiltonian
            auto hamiltonian = std::move(Hamiltonian_from_Config(configFile, geometry));
            // Spin System
            auto system = std::unique_ptr<Data::Spin_System>(new Data::Spin_System(std::move(hamiltonian),
                std::move(geometry), std::move(llg_params), std::move(mc_params), std::move(ema_params), std::move(mmf_params), false));

            Log(Log_Level::Info, Log_Sender::IO, "-------------- Spin System Initialised -------------");

            return system;
        }
        catch( ... )
        {
            spirit_handle_exception_core(fmt::format("Unable to initialize spin system from config file \"{}\"", configFile));
        }

        return nullptr;
    }// End Spin_System_from_Config


    void Bravais_Vectors_from_Config(const std::string configFile, std::vector<Vector3> & bravais_vectors, Data::BravaisLatticeType & bravais_lattice_type)
    {
        try
        {
            std::string bravais_lattice = "sc";
            // Manually specified bravais vectors/matrix?
            bool irregular = true;

            IO::Filter_File_Handle myfile(configFile);
            // Bravais lattice type or manually specified vectors/matrix
            if (myfile.Find("bravais_lattice"))
            {
                myfile.iss >> bravais_lattice;
                std::transform(bravais_lattice.begin(), bravais_lattice.end(), bravais_lattice.begin(), ::tolower);

                if (bravais_lattice == "sc")
                {
                    Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: simple cubic");
                    bravais_lattice_type = Data::BravaisLatticeType::SC;
                    bravais_vectors = Data::Geometry::BravaisVectorsSC();
                }
                else if (bravais_lattice == "fcc")
                {
                    Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: face-centered cubic");
                    bravais_lattice_type = Data::BravaisLatticeType::FCC;
                    bravais_vectors = Data::Geometry::BravaisVectorsFCC();
                }
                else if (bravais_lattice == "bcc")
                {
                    Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: body-centered cubic");
                    bravais_lattice_type = Data::BravaisLatticeType::BCC;
                    bravais_vectors = Data::Geometry::BravaisVectorsBCC();
                }
                else if (bravais_lattice == "hex2d")
                {
                    Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: hexagonal 2D (default: 60deg angle)");
                    bravais_lattice_type = Data::BravaisLatticeType::Hex2D;
                    bravais_vectors = Data::Geometry::BravaisVectorsHex2D60();
                }
                else if (bravais_lattice == "hex2d60")
                {
                    Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: hexagonal 2D 60deg angle");
                    bravais_lattice_type = Data::BravaisLatticeType::Hex2D;
                    bravais_vectors = Data::Geometry::BravaisVectorsHex2D60();
                }
                else if (bravais_lattice == "hex2d120")
                {
                    Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: hexagonal 2D 120deg angle");
                    bravais_lattice_type = Data::BravaisLatticeType::Hex2D;
                    bravais_vectors = Data::Geometry::BravaisVectorsHex2D120();
                }
                else
                    Log(Log_Level::Warning, Log_Sender::IO,
                        fmt::format("Bravais lattice \"{}\" unknown. Using simple cubic...", bravais_lattice));
            }
            else if (myfile.Find("bravais_vectors"))
            {
                Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: irregular");
                bravais_lattice_type = Data::BravaisLatticeType::Irregular;
                myfile.GetLine();
                myfile.iss >> bravais_vectors[0][0] >> bravais_vectors[0][1] >> bravais_vectors[0][2];
                myfile.GetLine();
                myfile.iss >> bravais_vectors[1][0] >> bravais_vectors[1][1] >> bravais_vectors[1][2];
                myfile.GetLine();
                myfile.iss >> bravais_vectors[2][0] >> bravais_vectors[2][1] >> bravais_vectors[2][2];

            }
            else if (myfile.Find("bravais_matrix"))
            {
                Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: irregular");
                bravais_lattice_type = Data::BravaisLatticeType::Irregular;
                myfile.GetLine();
                myfile.iss >> bravais_vectors[0][0] >> bravais_vectors[1][0] >> bravais_vectors[2][0];
                myfile.GetLine();
                myfile.iss >> bravais_vectors[0][1] >> bravais_vectors[1][1] >> bravais_vectors[2][1];
                myfile.GetLine();
                myfile.iss >> bravais_vectors[0][2] >> bravais_vectors[1][2] >> bravais_vectors[2][2];
            }
            else
                Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice not specified. Using simple cubic...");
        }
        catch( ... )
        {
            spirit_rethrow(	fmt::format("Unable to parse bravais vectors from config file \"{}\"", configFile) );
        }
    }// End Basis_from_Config

    std::shared_ptr<Data::Geometry> Geometry_from_Config(const std::string configFile)
    {
        try
        {
            //-------------- Insert default values here -----------------------------
            // Basis from separate file?
            std::string basis_file = "";
            // Bravais lattice type
            std::string bravais_lattice = "sc";
            // Bravais vectors {a, b, c}
            Data::BravaisLatticeType bravais_lattice_type = Data::BravaisLatticeType::SC;
            std::vector<Vector3> bravais_vectors = { Vector3{1,0,0}, Vector3{0,1,0}, Vector3{0,0,1} };
            // Atoms in the basis
            std::vector<Vector3> cell_atoms = { Vector3{0,0,0} };
            int n_cell_atoms = cell_atoms.size();
            // Basis cell composition information (atom types, magnetic moments, ...)
            Data::Basis_Cell_Composition cell_composition{ false, {0}, {0}, {1}, {} };
            // Lattice Constant [Angstrom]
            scalar lattice_constant = 1;
            // Number of translations nT for each basis direction
            intfield n_cells = { 100, 100, 1 };
            // Atom types
            field<Site> defect_sites(0);
            intfield    defect_types(0);
            int n_atom_types = 0;

            // Utility 1D array to build vectors and use Vectormath
            Vector3 build_array = { 0, 0, 0 };

            Log(Log_Level::Info, Log_Sender::IO, "Geometry: building");
            //------------------------------- Parser --------------------------------
            // iteration variables
            int iatom = 0, dim = 0;
            if( configFile != "" )
            {
                try
                {
                    Log(Log_Level::Info, Log_Sender::IO, "Reading Geometry Parameters");
                    IO::Filter_File_Handle myfile(configFile);

                    // Lattice constant
                    myfile.Read_Single(lattice_constant, "lattice_constant");

                    // Get the bravais lattice type and vectors
                    Bravais_Vectors_from_Config(configFile, bravais_vectors, bravais_lattice_type);

                    // Read basis cell
                    if (myfile.Find("basis"))
                    {
                        // Read number of atoms in the basis cell
                        myfile.GetLine();
                        myfile.iss >> n_cell_atoms;
                        cell_atoms = std::vector<Vector3>(n_cell_atoms);
                        cell_composition.iatom.resize(n_cell_atoms);
                        cell_composition.atom_type = std::vector<int>(n_cell_atoms, 0);
                        cell_composition.mu_s = std::vector<scalar>(n_cell_atoms, 1);

                        // Read atom positions
                        for (iatom = 0; iatom < n_cell_atoms; ++iatom)
                        {
                            myfile.GetLine();
                            myfile.iss >> cell_atoms[iatom][0] >> cell_atoms[iatom][1] >> cell_atoms[iatom][2];
                            cell_composition.iatom[iatom] = iatom;
                        }// endfor iatom
                    }

                    // Read number of basis cells
                    myfile.Read_3Vector(n_cells, "n_basis_cells");

                    // Defects
                    #ifdef SPIRIT_ENABLE_DEFECTS
                    int n_defects = 0;

                    std::string defectsFile = "";
                    if (myfile.Find("n_defects"))
                        defectsFile = configFile;
                    else if (myfile.Find("defects_from_file"))
                        myfile.iss >> defectsFile;

                    if (defectsFile.length() > 0)
                    {
                        // The file name should be valid so we try to read it
                        Defects_from_File(defectsFile, n_defects, defect_sites, defect_types);
                    }

                    // Disorder
                    if( myfile.Find("atom_types") )
                    {
                        myfile.iss >> n_atom_types;
                        cell_composition.disordered = true;
                        cell_composition.iatom.resize(n_atom_types);
                        cell_composition.atom_type.resize(n_atom_types);
                        cell_composition.mu_s.resize(n_atom_types);
                        cell_composition.concentration.resize(n_atom_types);
                        for (int itype = 0; itype < n_atom_types; ++itype)
                        {
                            myfile.GetLine();
                            myfile.iss >> cell_composition.iatom[itype];
                            myfile.iss >> cell_composition.atom_type[itype];
                            myfile.iss >> cell_composition.mu_s[itype];
                            myfile.iss >> cell_composition.concentration[itype];
                            // if ( !(myfile.iss >> mu_s[itype]) )
                            // {
                            //     Log(Log_Level::Warning, Log_Sender::IO,
                            //         fmt::format("Not enough values specified after 'mu_s'. Expected {}. Using mu_s[{}]=mu_s[0]={}", n_cell_atoms, iatom, mu_s[0]));
                            //     mu_s[iatom] = mu_s[0];
                            // }
                        }
                        Log(Log_Level::Warning, Log_Sender::IO,
                            fmt::format("{} atom types, iatom={} atom type={} concentration={}", n_atom_types, cell_composition.iatom[0], cell_composition.atom_type[0], cell_composition.concentration[0]));
                    }
                    #endif
                }// end try
                catch( ... )
                {
                    spirit_handle_exception_core(fmt::format("Failed to read Geometry parameters from file \"{}\". Leaving values at default.", configFile));
                }

                try
                {
                    IO::Filter_File_Handle myfile(configFile);

                    // Spin moment
                    if( !myfile.Find("atom_types") )
                    {
                        if( myfile.Find("mu_s") )
                        {
                            for (iatom = 0; iatom < n_cell_atoms; ++iatom)
                            {
                                if ( !(myfile.iss >> cell_composition.mu_s[iatom]) )
                                {
                                    Log(Log_Level::Warning, Log_Sender::IO, fmt::format(
                                        "Not enough values specified after 'mu_s'. Expected {}. Using mu_s[{}]=mu_s[0]={}",
                                        n_cell_atoms, iatom, cell_composition.mu_s[0]));
                                    cell_composition.mu_s[iatom] = cell_composition.mu_s[0];
                                }
                            }
                        }
                        else Log(Log_Level::Error, Log_Sender::IO, fmt::format("Keyword 'mu_s' not found. Using Default: {}", cell_composition.mu_s[0]));
                    }
                    // else
                    // {
                    //     cell_composition.mu_s = std::vector<scalar>(n_atom_types, 1);
                    //     if( myfile.Find("mu_s") )
                    //     {
                    //         for (int itype = 0; itype < n_atom_types; ++itype)
                    //         {
                    //             myfile.iss >> cell_composition.mu_s[itype];
                    //             // myfile.GetLine();
                    //             // myfile.iss >> cell_composition.iatom[itype];
                    //             // myfile.iss >> cell_composition.atom_type[itype];
                    //             // myfile.iss >> cell_composition.concentration[itype];
                    //         }
                    //     }
                    //     else Log(Log_Level::Error, Log_Sender::IO, fmt::format("Keyword 'mu_s' not found. Using Default: {}", cell_composition.mu_s[0]));
                    // }
                }// end try
                catch( ... )
                {
                    spirit_handle_exception_core(fmt::format("Unable to read mu_s from config file \"{}\"", configFile));
                }
            }// end if file=""
            else
                Log(Log_Level::Parameter, Log_Sender::IO, "Geometry: Using default configuration!");

            // Log the parameters
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Lattice constant = {} angstrom", lattice_constant));
            Log(Log_Level::Debug, Log_Sender::IO, "Bravais vectors in units of lattice constant");
            Log(Log_Level::Debug, Log_Sender::IO, fmt::format("        a = {}", bravais_vectors[0].transpose() / lattice_constant));
            Log(Log_Level::Debug, Log_Sender::IO, fmt::format("        b = {}", bravais_vectors[1].transpose() / lattice_constant));
            Log(Log_Level::Debug, Log_Sender::IO, fmt::format("        c = {}", bravais_vectors[2].transpose() / lattice_constant));
            Log(Log_Level::Parameter, Log_Sender::IO, "Bravais vectors");
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        a = {}", bravais_vectors[0].transpose()));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        b = {}", bravais_vectors[1].transpose()));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        c = {}", bravais_vectors[2].transpose()));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Basis cell: {}  atom(s)", n_cell_atoms));
            Log(Log_Level::Parameter, Log_Sender::IO, "Relative positions (first 10):");
            for( int iatom = 0; iatom < n_cell_atoms && iatom < 10; ++iatom )
                Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        atom {} at ({}), mu_s={}", iatom, cell_atoms[iatom].transpose(), cell_composition.mu_s[iatom]));

            Log(Log_Level::Parameter, Log_Sender::IO, "Absolute atom positions (first 10):", n_cell_atoms);
            for( int iatom = 0; iatom < n_cell_atoms && iatom < 10; ++iatom )
            {
                Vector3 cell_atom = lattice_constant * (
                      bravais_vectors[0] * cell_atoms[iatom][0]
                    + bravais_vectors[1] * cell_atoms[iatom][1]
                    + bravais_vectors[2] * cell_atoms[iatom][2] );
                Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        atom {} at ({})", iatom, cell_atom.transpose()));
            }

            if( cell_composition.disordered )
                Log(Log_Level::Parameter, Log_Sender::IO, "Note: the lattice has some disorder!");

            // Defects
            #ifdef SPIRIT_ENABLE_DEFECTS
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Geometry: {} defects. Printing the first 10:", defect_sites.size()));
            for (int i = 0; i < defect_sites.size(); ++i)
                if (i < 10) Log(Log_Level::Parameter, Log_Sender::IO, fmt::format(
                    "  defect[{}]: translations=({} {} {}), type=",
                    i, defect_sites[i].translations[0], defect_sites[i].translations[1], defect_sites[i].translations[2], defect_types[i]));
            #endif

            // Log parameters
            Log(Log_Level::Parameter, Log_Sender::IO, "Lattice: n_basis_cells");
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("       na = {}", n_cells[0]));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("       nb = {}", n_cells[1]));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("       nc = {}", n_cells[2]));

            // Pinning configuration
            auto pinning = Pinning_from_Config(configFile, cell_atoms.size());

            // Return geometry
            auto geometry = std::shared_ptr<Data::Geometry>(new
                Data::Geometry( bravais_vectors, n_cells, cell_atoms, cell_composition, lattice_constant,
                    pinning, {defect_sites, defect_types} ));

            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Geometry: {} spins", geometry->nos));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Geometry is {}-dimensional", geometry->dimensionality));
            Log(Log_Level::Info, Log_Sender::IO, "Geometry: built");
            return geometry;
        }
        catch( ... )
        {
            spirit_rethrow(	fmt::format("Unable to parse geometry from config file \"{}\"", configFile) );
        }

        return nullptr;
    }// end Geometry from Config

    Data::Pinning Pinning_from_Config(const std::string configFile, int n_cell_atoms)
    {
        //-------------- Insert default values here -----------------------------
        int na = 0, na_left = 0, na_right = 0;
        int nb = 0, nb_left = 0, nb_right = 0;
        int nc = 0, nc_left = 0, nc_right = 0;
        vectorfield pinned_cell(n_cell_atoms, Vector3{ 0,0,1 });
        // Additional pinned sites
        field<Site> pinned_sites(0);
        vectorfield pinned_spins(0);
        int n_pinned = 0;

        // Utility 1D array to build vectors and use Vectormath
        Vector3 build_array = { 0, 0, 0 };

        #ifdef SPIRIT_ENABLE_PINNING
            Log(Log_Level::Info, Log_Sender::IO, "Reading Pinning Configuration");
            //------------------------------- Parser --------------------------------
            if( configFile != "" )
            {
                try
                {
                    IO::Filter_File_Handle myfile(configFile);

                    // N_a
                    myfile.Read_Single(na_left, "pin_na_left", false);
                    myfile.Read_Single(na_right, "pin_na_right", false);
                    myfile.Read_Single(na, "pin_na ", false);
                    if (na > 0 && (na_left == 0 || na_right == 0))
                    {
                        na_left = na;
                        na_right = na;
                    }

                    // N_b
                    myfile.Read_Single(nb_left, "pin_nb_left", false);
                    myfile.Read_Single(nb_right, "pin_nb_right", false);
                    myfile.Read_Single(nb, "pin_nb ", false);
                    if (nb > 0 && (nb_left == 0 || nb_right == 0))
                    {
                        nb_left = nb;
                        nb_right = nb;
                    }

                    // N_c
                    myfile.Read_Single(nc_left, "pin_nc_left", false);
                    myfile.Read_Single(nc_right, "pin_nc_right", false);
                    myfile.Read_Single(nc, "pin_nc ", false);
                    if (nc > 0 && (nc_left == 0 || nc_right == 0))
                    {
                        nc_left = nc;
                        nc_right = nc;
                    }

                    // How should the cells be pinned
                    if (na_left > 0 || na_right > 0 ||
                        nb_left > 0 || nb_right > 0 ||
                        nc_left > 0 || nc_right > 0)
                    {
                        if (myfile.Find("pinning_cell"))
                        {
                            for (int i = 0; i < n_cell_atoms; ++i)
                            {
                                myfile.GetLine();
                                myfile.iss >> pinned_cell[i][0] >> pinned_cell[i][1] >> pinned_cell[i][2];
                            }
                        }
                        else
                        {
                            na_left = 0; na_right = 0;
                            nb_left = 0; nb_right = 0;
                            nc_left = 0; nc_right = 0;
                            Log(Log_Level::Warning, Log_Sender::IO, "Pinning specified, but keyword 'pinning_cell' not found. Won't pin any spins!");
                        }
                    }

                    // Additional pinned sites
                    std::string pinnedFile = "";
                    if (myfile.Find("n_pinned"))
                        pinnedFile = configFile;
                    else if (myfile.Find("pinned_from_file"))
                        myfile.iss >> pinnedFile;

                    if(pinnedFile != "")
                    {
                        // The file name should be valid so we try to read it
                        Pinned_from_File(pinnedFile, n_pinned, pinned_sites, pinned_spins);
                    }
                    else Log(Log_Level::Parameter, Log_Sender::IO, "wtf no pinnedFile");

                }// end try
                catch( ... )
                {
                    spirit_handle_exception_core(fmt::format("Failed to read Pinning from file \"{}\". Leaving values at default.", configFile));
                }

            }// end if file=""
            else Log(Log_Level::Parameter, Log_Sender::IO, "No pinning");

            // Create Pinning
            auto pinning = Data::Pinning{
                na_left, na_right,
                nb_left, nb_right,
                nc_left, nc_right,
                pinned_cell,
                pinned_sites, pinned_spins};


            // Return Pinning
            Log(Log_Level::Parameter, Log_Sender::IO, "Pinning:");
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        n_a: left={}, right={}", na_left, na_right));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        n_b: left={}, right={}", nb_left, nb_right));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        n_c: left={}, right={}", nc_left, nc_right));
            for (int i = 0; i < n_cell_atoms; ++i)
                Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        cell atom[{}]      = ({})", i, pinned_cell[0].transpose()));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {} additional pinned sites. Showing the first 10:", n_pinned));
            for (int i = 0; i < n_pinned; ++i)
            {
                if( i<10 )
                    Log(Log_Level::Parameter, Log_Sender::IO, fmt::format(
                        "             pinned site[{}]: {} at ({} {} {}) = ({})",
                        i, pinned_sites[i].i, pinned_sites[i].translations[0], pinned_sites[i].translations[1], pinned_sites[i].translations[2], pinned_spins[0].transpose()));
            }
            Log(Log_Level::Info, Log_Sender::IO, "Pinning: read");
            return pinning;
        #else // SPIRIT_ENABLE_PINNING
            Log(Log_Level::Info, Log_Sender::IO, "Pinning is disabled");
            if( configFile != "" )
            {
                try
                {
                    IO::Filter_File_Handle myfile(configFile);
                    if (myfile.Find("pinning_cell"))
                        Log(Log_Level::Warning, Log_Sender::IO, "You specified a pinning cell even though pinning is disabled!");
                }
                catch( ... )
                {
                    spirit_handle_exception_core(fmt::format("Failed to read pinning parameters from file \"{}\". Leaving values at default.", configFile));
                }
            }

            return Data::Pinning{
                0, 0, 0, 0, 0, 0,
                vectorfield(0),
                field<Site>(0),
                vectorfield(0) };
        #endif // SPIRIT_ENABLE_PINNING
    }

    std::unique_ptr<Data::Parameters_Method_LLG> Parameters_Method_LLG_from_Config(const std::string configFile)
    {
        // Default parameters
        auto parameters = std::unique_ptr<Data::Parameters_Method_LLG>(new Data::Parameters_Method_LLG());

        // PRNG Seed
        std::srand((unsigned int)std::time(0));
        parameters->rng_seed = std::rand();

        // Maximum wall time
        std::string str_max_walltime = "0";

        // Configuration output filetype
        int output_configuration_filetype = (int)parameters->output_vf_filetype;

        // Parse
        Log(Log_Level::Info, Log_Sender::IO, "Parameters LLG: building");
        if( configFile != "" )
        {
            try
            {
                IO::Filter_File_Handle myfile(configFile);

                // Output parameters
                myfile.Read_Single(parameters->output_file_tag,"output_file_tag");
                myfile.Read_Single(parameters->output_folder,  "llg_output_folder");
                myfile.Read_Single(parameters->output_any,     "llg_output_any");
                myfile.Read_Single(parameters->output_initial, "llg_output_initial");
                myfile.Read_Single(parameters->output_final,   "llg_output_final");
                myfile.Read_Single(parameters->output_energy_spin_resolved,         "llg_output_energy_spin_resolved");
                myfile.Read_Single(parameters->output_energy_step,                  "llg_output_energy_step");
                myfile.Read_Single(parameters->output_energy_archive,               "llg_output_energy_archive");
                myfile.Read_Single(parameters->output_energy_divide_by_nspins,      "llg_output_energy_divide_by_nspins");
                myfile.Read_Single(parameters->output_energy_add_readability_lines, "llg_output_energy_add_readability_lines");
                myfile.Read_Single(parameters->output_configuration_step,           "llg_output_configuration_step");
                myfile.Read_Single(parameters->output_configuration_archive,        "llg_output_configuration_archive");
                myfile.Read_Single(output_configuration_filetype,                   "llg_output_configuration_filetype");
                parameters->output_vf_filetype = IO::VF_FileFormat(output_configuration_filetype);
                // Method parameters
                myfile.Read_Single(str_max_walltime, "llg_max_walltime");
                parameters->max_walltime_sec = (long int)Utility::Timing::DurationFromString(str_max_walltime).count();
                myfile.Read_Single(parameters->rng_seed, "llg_seed");
                myfile.Read_Single(parameters->n_iterations, "llg_n_iterations");
                myfile.Read_Single(parameters->n_iterations_log, "llg_n_iterations_log");
                myfile.Read_Single(parameters->dt, "llg_dt");
                myfile.Read_Single(parameters->temperature, "llg_temperature");
                myfile.Read_Vector3(parameters->temperature_gradient_direction, "llg_temperature_gradient_direction");
                parameters->temperature_gradient_direction.normalize();
                myfile.Read_Single(parameters->temperature_gradient_inclination, "llg_temperature_gradient_inclination");
                myfile.Read_Single(parameters->damping, "llg_damping");
                myfile.Read_Single(parameters->beta, "llg_beta");
                // myfile.Read_Single(parameters->renorm_sd, "llg_renorm");
                myfile.Read_Single(parameters->stt_use_gradient, "llg_stt_use_gradient");
                myfile.Read_Single(parameters->stt_magnitude, "llg_stt_magnitude");
                myfile.Read_Vector3(parameters->stt_polarisation_normal, "llg_stt_polarisation_normal");
                parameters->stt_polarisation_normal.normalize();
                myfile.Read_Single(parameters->force_convergence, "llg_force_convergence");
            }
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format(
                    "Unable to parse LLG parameters from config file \"{}\"", configFile));
            }
        }
        else
            Log(Log_Level::Parameter, Log_Sender::IO, "Parameters LLG: Using default configuration!");

        // Return
        Log(Log_Level::Parameter, Log_Sender::IO, "Parameters LLG:");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "seed", parameters->rng_seed));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "time step [ps]", parameters->dt));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "temperature [K]", parameters->temperature));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "temperature gradient direction", parameters->temperature_gradient_direction.transpose()));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "temperature gradient inclination", parameters->temperature_gradient_inclination));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "damping", parameters->damping));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "beta", parameters->beta));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "stt use gradient", parameters->stt_use_gradient));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "stt magnitude", parameters->stt_magnitude));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "stt normal", parameters->stt_polarisation_normal.transpose()));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {:e}", "force convergence", parameters->force_convergence));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "maximum walltime", str_max_walltime));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "n_iterations", parameters->n_iterations));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "n_iterations_log", parameters->n_iterations_log));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = \"{}\"", "output_folder", parameters->output_folder));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "output_any", parameters->output_any));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "output_initial", parameters->output_initial));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "output_final", parameters->output_final));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_step", parameters->output_energy_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_archive", parameters->output_energy_archive));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_spin_resolved", parameters->output_energy_spin_resolved));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_add_readability_lines", parameters->output_energy_add_readability_lines));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_configuration_step", parameters->output_configuration_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_configuration_filetype", (int)parameters->output_vf_filetype));

        Log(Log_Level::Info, Log_Sender::IO, "Parameters LLG: built");
        return parameters;
    }// end Parameters_Method_LLG_from_Config

    std::unique_ptr<Data::Parameters_Method_EMA> Parameters_Method_EMA_from_Config(const std::string configFile)
    {
        // Default parameters
        auto parameters = std::unique_ptr<Data::Parameters_Method_EMA>(new Data::Parameters_Method_EMA());

        // Maximum wall time
        std::string str_max_walltime = "0";

        // Parse
        Log(Log_Level::Info, Log_Sender::IO, "Parameters EMA: building");
        if( configFile != "" )
        {
            try
            {
                IO::Filter_File_Handle myfile(configFile);

                // Output parameters
                myfile.Read_Single(parameters->output_folder,  "ema_output_folder");
                myfile.Read_Single(parameters->output_file_tag,"output_file_tag");
                myfile.Read_Single(parameters->output_any,     "ema_output_any");
                myfile.Read_Single(parameters->output_initial, "ema_output_initial");
                myfile.Read_Single(parameters->output_final,   "ema_output_final");
                myfile.Read_Single(parameters->output_energy_divide_by_nspins, "ema_output_energy_divide_by_nspins");
                myfile.Read_Single(parameters->output_energy_spin_resolved,    "ema_output_energy_spin_resolved");
                myfile.Read_Single(parameters->output_energy_step,             "ema_output_energy_step");
                myfile.Read_Single(parameters->output_energy_archive,          "ema_output_energy_archive");
                myfile.Read_Single(parameters->output_configuration_step,      "ema_output_configuration_step");
                myfile.Read_Single(parameters->output_configuration_archive,   "ema_output_configuration_archive");
                // Method parameters
                myfile.Read_Single(str_max_walltime, "ema_max_walltime");
                parameters->max_walltime_sec = (long int)Utility::Timing::DurationFromString(str_max_walltime).count();
                myfile.Read_Single(parameters->n_iterations, "ema_n_iterations");
                myfile.Read_Single(parameters->n_iterations_log, "ema_n_iterations_log");
                myfile.Read_Single(parameters->n_modes, "ema_n_modes");
                myfile.Read_Single(parameters->n_mode_follow, "ema_n_mode_follow");
                myfile.Read_Single(parameters->frequency, "ema_frequency");
                myfile.Read_Single(parameters->amplitude, "ema_amplitude");
            }
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format(
                    "Unable to parse EMA parameters from config file \"{}\"", configFile));
            }
        }
        else
            Log(Log_Level::Parameter, Log_Sender::IO, "Parameters EMA: Using default configuration!");

        // Return
        Log(Log_Level::Parameter, Log_Sender::IO, "Parameters EMA:");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "n_modes", parameters->n_modes));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "n_mode_follow", parameters->n_mode_follow));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "frequency", parameters->frequency));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "amplitude", parameters->amplitude));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "n_iterations_log", parameters->n_iterations_log));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "n_iterations", parameters->n_iterations));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "maximum walltime", str_max_walltime));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_configuration_step", parameters->output_configuration_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_archive", parameters->output_energy_archive));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_step", parameters->output_energy_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_spin_resolved", parameters->output_energy_spin_resolved));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "output_final", parameters->output_final));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "output_initial", parameters->output_initial));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "output_any", parameters->output_any));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = \"{}\"", "output_folder", parameters->output_folder));

        Log(Log_Level::Info, Log_Sender::IO, "Parameters EMA: built");
        return parameters;
    }

    std::unique_ptr<Data::Parameters_Method_MC> Parameters_Method_MC_from_Config(const std::string configFile)
    {
        // Default parameters
        auto parameters = std::unique_ptr<Data::Parameters_Method_MC>(new Data::Parameters_Method_MC());

        // PRNG Seed
        std::srand((unsigned int)std::time(0));
        parameters->rng_seed = std::rand();

        // Maximum wall time
        std::string str_max_walltime = "0";

        // Configuration output filetype
        int output_configuration_filetype = (int)parameters->output_vf_filetype;

        // Parse
        Log(Log_Level::Info, Log_Sender::IO, "Parameters MC: building");
        if( configFile != "" )
        {
            try
            {
                IO::Filter_File_Handle myfile(configFile);

                // Output parameters
                myfile.Read_Single(parameters->output_file_tag, "output_file_tag");
                myfile.Read_Single(parameters->output_folder,   "mc_output_folder");
                myfile.Read_Single(parameters->output_any,      "mc_output_any");
                myfile.Read_Single(parameters->output_initial,  "mc_output_initial");
                myfile.Read_Single(parameters->output_final,    "mc_output_final");
                myfile.Read_Single(parameters->output_energy_spin_resolved,     "mc_output_energy_spin_resolved");
                myfile.Read_Single(parameters->output_energy_step,              "mc_output_energy_step");
                myfile.Read_Single(parameters->output_energy_archive,           "mc_output_energy_archive");
                myfile.Read_Single(parameters->output_energy_divide_by_nspins,  "mc_output_energy_divide_by_nspins");
                myfile.Read_Single(parameters->output_energy_add_readability_lines, "mc_output_energy_add_readability_lines");
                myfile.Read_Single(parameters->output_configuration_step,       "mc_output_configuration_step");
                myfile.Read_Single(parameters->output_configuration_archive,    "mc_output_configuration_archive");
                myfile.Read_Single(output_configuration_filetype,               "mc_output_configuration_filetype");
                parameters->output_vf_filetype = IO::VF_FileFormat(output_configuration_filetype);
                // Method parameters
                myfile.Read_Single(str_max_walltime, "mc_max_walltime");
                parameters->max_walltime_sec = (long int)Utility::Timing::DurationFromString(str_max_walltime).count();
                myfile.Read_Single(parameters->rng_seed, "mc_seed");
                myfile.Read_Single(parameters->n_iterations, "mc_n_iterations");
                myfile.Read_Single(parameters->n_iterations_log, "mc_n_iterations_log");
                myfile.Read_Single(parameters->temperature, "mc_temperature");
                myfile.Read_Single(parameters->acceptance_ratio_target, "mc_acceptance_ratio");
            }
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format(
                    "Unable to parse MC parameters from config file \"{}\"", configFile));
            }
        }
        else
            Log(Log_Level::Parameter, Log_Sender::IO, "Parameters MC: Using default configuration!");

        // Return
        Log(Log_Level::Parameter, Log_Sender::IO, "Parameters MC:");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "seed", parameters->rng_seed));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "temperature", parameters->temperature));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "acceptance_ratio", parameters->acceptance_ratio_target));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "maximum walltime", str_max_walltime));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "n_iterations", parameters->n_iterations));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "n_iterations_log", parameters->n_iterations_log));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = \"{}\"", "output_folder", parameters->output_folder));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "output_any", parameters->output_any));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "output_initial", parameters->output_initial));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "output_final", parameters->output_final));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_step", parameters->output_energy_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_archive", parameters->output_energy_archive));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_spin_resolved", parameters->output_energy_spin_resolved));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_add_readability_lines", parameters->output_energy_add_readability_lines));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_configuration_step", parameters->output_configuration_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_configuration_filetype", (int)parameters->output_vf_filetype));

        Log(Log_Level::Info, Log_Sender::IO, "Parameters MC: built");
        return parameters;
    }

    std::unique_ptr<Data::Parameters_Method_GNEB> Parameters_Method_GNEB_from_Config(const std::string configFile)
    {
        // Default parameters
        auto parameters = std::unique_ptr<Data::Parameters_Method_GNEB>(new Data::Parameters_Method_GNEB());

        // Maximum wall time
        std::string str_max_walltime = "0";

        // Chain output filetype
        int output_chain_filetype = (int)parameters->output_vf_filetype;

        // Parse
        Log(Log_Level::Info, Log_Sender::IO, "Parameters GNEB: building");
        if( configFile != "" )
        {
            try
            {
                IO::Filter_File_Handle myfile(configFile);

                // Output parameters
                myfile.Read_Single(parameters->output_file_tag, "output_file_tag");
                myfile.Read_Single(parameters->output_folder,   "gneb_output_folder");
                myfile.Read_Single(parameters->output_any,      "gneb_output_any");
                myfile.Read_Single(parameters->output_initial,  "gneb_output_initial");
                myfile.Read_Single(parameters->output_final,    "gneb_output_final");
                myfile.Read_Single(parameters->output_energies_step,                "gneb_output_energies_step");
                myfile.Read_Single(parameters->output_energies_add_readability_lines, "gneb_output_energies_add_readability_lines");
                myfile.Read_Single(parameters->output_energies_interpolated,        "gneb_output_energies_interpolated");
                myfile.Read_Single(parameters->output_energies_divide_by_nspins,    "gneb_output_energies_divide_by_nspins");
                myfile.Read_Single(parameters->output_chain_step,                   "gneb_output_chain_step");
                myfile.Read_Single(output_chain_filetype,                           "gneb_output_chain_filetype");
                parameters->output_vf_filetype = IO::VF_FileFormat(output_chain_filetype);
                // Method parameters
                myfile.Read_Single(str_max_walltime, "gneb_max_walltime");
                parameters->max_walltime_sec = (long int)Utility::Timing::DurationFromString(str_max_walltime).count();
                myfile.Read_Single(parameters->spring_constant, "gneb_spring_constant");
                myfile.Read_Single(parameters->force_convergence, "gneb_force_convergence");
                myfile.Read_Single(parameters->n_iterations, "gneb_n_iterations");
                myfile.Read_Single(parameters->n_iterations_log, "gneb_n_iterations_log");
                myfile.Read_Single(parameters->n_E_interpolations, "gneb_n_energy_interpolations");
            }
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format(
                    "Unable to parse GNEB parameters from config file \"{}\"", configFile));
            }
        }
        else
            Log(Log_Level::Parameter, Log_Sender::IO, "Parameters GNEB: Using default configuration!");

        // Return
        Log(Log_Level::Parameter, Log_Sender::IO, "Parameters GNEB:");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<18} = {}", "spring_constant", parameters->spring_constant));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<18} = {}", "n_E_interpolations", parameters->n_E_interpolations));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<18} = {:e}", "force convergence", parameters->force_convergence));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<18} = {}", "maximum walltime", str_max_walltime));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<18} = {}", "n_iterations", parameters->n_iterations));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<18} = {}", "n_iterations_log", parameters->n_iterations_log));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<18} = \"{}\"", "output_folder", parameters->output_folder));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<18} = {}", "output_any", parameters->output_any));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<18} = {}", "output_initial", parameters->output_initial));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<18} = {}", "output_final", parameters->output_final));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<18} = {}", "output_energies_step", parameters->output_energies_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<18} = {}", "output_energies_add_readability_lines", parameters->output_energies_add_readability_lines));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<18} = {}", "output_chain_step", parameters->output_chain_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<18} = {}", "output_chain_filetype", (int)parameters->output_vf_filetype));

        Log(Log_Level::Info, Log_Sender::IO, "Parameters GNEB: built");
        return parameters;
    }// end Parameters_Method_LLG_from_Config

    std::unique_ptr<Data::Parameters_Method_MMF> Parameters_Method_MMF_from_Config(const std::string configFile)
    {
        // Default parameters
        auto parameters = std::unique_ptr<Data::Parameters_Method_MMF>(new Data::Parameters_Method_MMF());

        // Maximum wall time
        std::string str_max_walltime = "0";

        // Configuration output filetype
        int output_configuration_filetype = (int)parameters->output_vf_filetype;

        // Parse
        Log(Log_Level::Info, Log_Sender::IO, "Parameters MMF: building");
        if( configFile != "" )
        {
            try
            {
                IO::Filter_File_Handle myfile(configFile);

                // Output parameters
                myfile.Read_Single(parameters->output_file_tag, "output_file_tag");
                myfile.Read_Single(parameters->output_folder,   "mmf_output_folder");
                myfile.Read_Single(parameters->output_any,      "mmf_output_any");
                myfile.Read_Single(parameters->output_initial,  "mmf_output_initial");
                myfile.Read_Single(parameters->output_final,    "mmf_output_final");
                myfile.Read_Single(parameters->output_energy_step,                  "mmf_output_energy_step");
                myfile.Read_Single(parameters->output_energy_archive,               "mmf_output_energy_archive");
                myfile.Read_Single(parameters->output_energy_divide_by_nspins,      "mmf_output_energy_divide_by_nspins");
                myfile.Read_Single(parameters->output_energy_add_readability_lines, "mmf_output_energy_add_readability_lines");
                myfile.Read_Single(parameters->output_configuration_step,           "mmf_output_configuration_step");
                myfile.Read_Single(parameters->output_configuration_archive,        "mmf_output_configuration_archive");
                myfile.Read_Single(output_configuration_filetype,                   "mmf_output_configuration_filetype");
                parameters->output_vf_filetype = IO::VF_FileFormat(output_configuration_filetype);
                // Method parameters
                myfile.Read_Single(str_max_walltime,  "mmf_max_walltime");
                parameters->max_walltime_sec = (long int)Utility::Timing::DurationFromString(str_max_walltime).count();
                myfile.Read_Single(parameters->force_convergence, "mmf_force_convergence");
                myfile.Read_Single(parameters->n_iterations,      "mmf_n_iterations");
                myfile.Read_Single(parameters->n_iterations_log,  "mmf_n_iterations_log");
                myfile.Read_Single(parameters->n_modes,           "mmf_n_modes");
                myfile.Read_Single(parameters->n_mode_follow,     "mmf_n_mode_follow");
            }
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format(
                    "Unable to parse MMF parameters from config file \"{}\"", configFile));
            }
        }
        else
            Log(Log_Level::Parameter, Log_Sender::IO, "Parameters MMF: Using default configuration!");

        // Return
        Log(Log_Level::Parameter, Log_Sender::IO, "Parameters MMF:");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {:e}", "force convergence", parameters->force_convergence));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "maximum walltime", str_max_walltime));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "n_iterations", parameters->n_iterations));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "n_iterations_log", parameters->n_iterations_log));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = \"{}\"", "output_folder", parameters->output_folder));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "output_any", parameters->output_any));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "output_initial", parameters->output_initial));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<17} = {}", "output_final", parameters->output_final));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_step", parameters->output_energy_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_archive", parameters->output_energy_archive));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_energy_add_readability_lines", parameters->output_energy_add_readability_lines));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_configuration_step", parameters->output_configuration_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<30} = {}", "output_configuration_filetype", (int)parameters->output_vf_filetype));

        Log(Log_Level::Info, Log_Sender::IO, "Parameters MMF: built");
        return parameters;
    }

    std::unique_ptr<Engine::Hamiltonian> Hamiltonian_from_Config(const std::string configFile, std::shared_ptr<Data::Geometry> geometry)
    {
        //-------------- Insert default values here -----------------------------
        // The type of hamiltonian we will use
        std::string hamiltonian_type = "heisenberg_neighbours";

        //------------------------------- Parser --------------------------------
        Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian: building");

        // Hamiltonian type
        if( configFile != "" )
        {
            try
            {
                Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian: deciding type");
                IO::Filter_File_Handle myfile(configFile);

                // What hamiltonian do we use?
                myfile.Read_Single(hamiltonian_type, "hamiltonian");
            }// end try
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Unable to read Hamiltonian type from config file \"{}\". Using default.", configFile));
                hamiltonian_type = "heisenberg_neighbours";
            }
        }
        else
            Log(Log_Level::Parameter, Log_Sender::IO, "Hamiltonian: Using default Hamiltonian: " + hamiltonian_type);

        // Hamiltonian
        std::unique_ptr<Engine::Hamiltonian> hamiltonian;
        try
        {
            if( hamiltonian_type == "heisenberg_neighbours" || hamiltonian_type == "heisenberg_pairs" )
                hamiltonian = Hamiltonian_Heisenberg_from_Config(configFile, geometry, hamiltonian_type);
            else if( hamiltonian_type == "micromagnetic" )
                hamiltonian = std::move(Hamiltonian_Micromagnetic_from_Config(configFile, geometry));
            else if( hamiltonian_type == "gaussian" )
                hamiltonian = std::move(Hamiltonian_Gaussian_from_Config(configFile, geometry));
            else
                spirit_throw(Exception_Classifier::System_not_Initialized, Log_Level::Severe, fmt::format("Hamiltonian: Invalid type \"{}\"", hamiltonian_type));
        }
        catch( ... )
        {
            spirit_handle_exception_core(fmt::format("Unable to initialize Hamiltonian from config file \"{}\"", configFile));
        }

        // Return
        Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian: built hamiltonian of type: " + hamiltonian_type);
        return hamiltonian;
    }

    std::unique_ptr<Engine::Hamiltonian_Heisenberg> Hamiltonian_Heisenberg_from_Config(const std::string configFile, std::shared_ptr<Data::Geometry> geometry, std::string hamiltonian_type)
    {
        //-------------- Insert default values here -----------------------------
        // Boundary conditions (a, b, c)
        std::vector<int> boundary_conditions_i = { 0, 0, 0 };
        intfield boundary_conditions = { false, false, false };

        // External Magnetic Field
        scalar B = 0;
        Vector3 B_normal = { 0.0, 0.0, 1.0 };

        // Anisotropy
        std::string anisotropy_file = "";
        scalar K = 0;
        Vector3 K_normal = { 0.0, 0.0, 1.0 };
        bool anisotropy_from_file = false;
        intfield    anisotropy_index(geometry->n_cell_atoms);
        scalarfield anisotropy_magnitude(geometry->n_cell_atoms, 0.0);
        vectorfield anisotropy_normal(geometry->n_cell_atoms, K_normal);

        // ------------ Pair Interactions ------------
        int n_pairs = 0;
        std::string interaction_pairs_file = "";
        bool interaction_pairs_from_file = false;
        pairfield exchange_pairs(0); scalarfield exchange_magnitudes(0);
        pairfield dmi_pairs(0); scalarfield dmi_magnitudes(0); vectorfield dmi_normals(0);

        // Number of shells in which we calculate neighbours
        int n_shells_exchange = exchange_magnitudes.size();
        // DM constant
        int n_shells_dmi = dmi_magnitudes.size();
        int dm_chirality = 1;

        std::string ddi_method_str = "none";
        auto ddi_method = Engine::DDI_Method::None;
        intfield ddi_n_periodic_images = { 4, 4, 4 };
        scalar ddi_radius = 0.0;

        // ------------ Quadruplet Interactions ------------
        int n_quadruplets = 0;
        std::string quadruplets_file = "";
        bool quadruplets_from_file = false;
        quadrupletfield quadruplets(0); scalarfield quadruplet_magnitudes(0);

        //------------------------------- Parser --------------------------------
        Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Heisenberg: building");
        // iteration variables
        int iatom = 0;
        if( configFile != "" )
        {
            try
            {
                IO::Filter_File_Handle myfile(configFile);

                // Boundary conditions
                myfile.Read_3Vector(boundary_conditions_i, "boundary_conditions");
                boundary_conditions[0] = (boundary_conditions_i[0] != 0);
                boundary_conditions[1] = (boundary_conditions_i[1] != 0);
                boundary_conditions[2] = (boundary_conditions_i[2] != 0);
            }// end try
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Unable to read boundary conditions from config file \"{}\"", configFile));
            }


            try
            {
                IO::Filter_File_Handle myfile(configFile);

                // Read parameters from config if available
                myfile.Read_Single(B, "external_field_magnitude");
                myfile.Read_Vector3(B_normal, "external_field_normal");
                B_normal.normalize();
                if (B_normal.norm() < 1e-8)
                {
                    B_normal = { 0,0,1 };
                    Log(Log_Level::Warning, Log_Sender::IO, "Input for 'external_field_normal' had norm zero and has been set to (0,0,1)");
                }
            }// end try
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Unable to read external field from config file \"{}\"", configFile));
            }

            try
            {
                IO::Filter_File_Handle myfile(configFile);

                // Anisotropy
                if (myfile.Find("n_anisotropy"))
                    anisotropy_file = configFile;
                else if (myfile.Find("anisotropy_file"))
                    myfile.iss >> anisotropy_file;
                if (anisotropy_file.length() > 0)
                {
                    // The file name should be valid so we try to read it
                    Anisotropy_from_File(anisotropy_file, geometry, n_pairs,
                        anisotropy_index, anisotropy_magnitude, anisotropy_normal);

                    anisotropy_from_file = true;
                    if (anisotropy_index.size() != 0)
                    {
                        K = anisotropy_magnitude[0];
                        K_normal = anisotropy_normal[0];
                    }
                    else
                    {
                        K = 0;
                        K_normal = { 0,0,0 };
                    }
                }
                else
                {
                    // Read parameters from config
                    myfile.Read_Single(K, "anisotropy_magnitude");
                    myfile.Read_Vector3(K_normal, "anisotropy_normal");
                    K_normal.normalize();

                    if (K != 0)
                    {
                        // Fill the arrays
                        for (int i = 0; i < anisotropy_index.size(); ++i)
                        {
                            anisotropy_index[i] = i;
                            anisotropy_magnitude[i] = K;
                            anisotropy_normal[i] = K_normal;
                        }
                    }
                    else
                    {
                        anisotropy_index = intfield(0);
                        anisotropy_magnitude = scalarfield(0);
                        anisotropy_normal = vectorfield(0);
                    }
                }
            }// end try
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Unable to read anisotropy from config file \"{}\"", configFile));
            }

            if (hamiltonian_type == "heisenberg_pairs")
            {
                try
                {
                    IO::Filter_File_Handle myfile(configFile);

                    // Interaction Pairs
                    if (myfile.Find("n_interaction_pairs"))
                        interaction_pairs_file = configFile;
                    else if (myfile.Find("interaction_pairs_file"))
                        myfile.iss >> interaction_pairs_file;

                    if (interaction_pairs_file.length() > 0)
                    {
                        // The file name should be valid so we try to read it
                        Pairs_from_File(interaction_pairs_file, geometry, n_pairs,
                            exchange_pairs, exchange_magnitudes,
                            dmi_pairs, dmi_magnitudes, dmi_normals);
                    }
                    //else
                    //{
                    //	Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_Heisenberg: Default Interaction pairs have not been implemented yet.");
                    //	throw Exception::System_not_Initialized;
                    //	// Not implemented!
                    //}
                }// end try
                catch( ... )
                {
                    spirit_handle_exception_core(fmt::format("Unable to read interaction pairs from config file \"{}\"", configFile));
                }
            }
            else
            {
                try
                {
                    IO::Filter_File_Handle myfile(configFile);

                    myfile.Read_Single(n_shells_exchange, "n_shells_exchange");
                    if (exchange_magnitudes.size() != n_shells_exchange)
                        exchange_magnitudes = scalarfield(n_shells_exchange);
                    if (n_shells_exchange > 0)
                    {
                        if (myfile.Find("jij"))
                        {
                            for (int ishell = 0; ishell < n_shells_exchange; ++ishell)
                                myfile.iss >> exchange_magnitudes[ishell];
                        }
                        else
                            Log(Log_Level::Warning, Log_Sender::IO, fmt::format(
                                "Hamiltonian_Heisenberg: Keyword 'jij' not found. Using Default:  {}", exchange_magnitudes[0]));
                    }
                }// end try
                catch( ... )
                {
                    spirit_handle_exception_core(fmt::format("Failed to read exchange parameters from config file \"{}\"", configFile));
                }

                try
                {
                    IO::Filter_File_Handle myfile(configFile);

                    myfile.Read_Single(n_shells_dmi, "n_shells_dmi");
                    if (dmi_magnitudes.size() != n_shells_dmi)
                        dmi_magnitudes = scalarfield(n_shells_dmi);
                    if (n_shells_dmi > 0)
                    {
                        if (myfile.Find("dij"))
                        {
                            for (int ishell = 0; ishell < n_shells_dmi; ++ishell)
                                myfile.iss >> dmi_magnitudes[ishell];
                        }
                        else
                            Log(Log_Level::Warning, Log_Sender::IO, fmt::format(
                                "Hamiltonian_Heisenberg: Keyword 'dij' not found. Using Default:  {}", dmi_magnitudes[0]));
                    }
                    myfile.Read_Single(dm_chirality, "dm_chirality");

                }// end try
                catch( ... )
                {
                    spirit_handle_exception_core(fmt::format("Failed to read DMI parameters from config file \"{}\"", configFile));
                }
            }

            try
            {
                IO::Filter_File_Handle myfile(configFile);

                // DDI method
                myfile.Read_String(ddi_method_str, "ddi_method");
                if( ddi_method_str == "none" )
                    ddi_method = Engine::DDI_Method::None;
                else if( ddi_method_str == "fft" )
                    ddi_method = Engine::DDI_Method::FFT;
                else if( ddi_method_str == "fmm" )
                    ddi_method = Engine::DDI_Method::FMM;
                else if( ddi_method_str == "cutoff" )
                    ddi_method = Engine::DDI_Method::Cutoff;
                else
                {
                    Log(Log_Level::Warning, Log_Sender::IO, fmt::format(
                        "Hamiltonian_Heisenberg: Keyword 'ddi_method' got passed invalid method \"{}\". Setting to \"none\".", ddi_method_str));
                    ddi_method_str = "none";
                }

                // Number of periodical images
                myfile.Read_3Vector(ddi_n_periodic_images, "ddi_n_periodic_images");
                // myfile.Read_Single(ddi_n_periodic_images, "ddi_n_periodic_images");

                // Dipole-dipole cutoff radius
                myfile.Read_Single(ddi_radius, "ddi_radius");
            }// end try
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Unable to read DDI radius from config file \"{}\"", configFile));
            }

            try
            {
                IO::Filter_File_Handle myfile(configFile);

                // Interaction Quadruplets
                if (myfile.Find("n_interaction_quadruplets"))
                    quadruplets_file = configFile;
                else if (myfile.Find("interaction_quadruplets_file"))
                    myfile.iss >> quadruplets_file;

                if (quadruplets_file.length() > 0)
                {
                    // The file name should be valid so we try to read it
                    Quadruplets_from_File(quadruplets_file, geometry, n_quadruplets,
                        quadruplets, quadruplet_magnitudes);
                }

            }// end try
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Unable to read interaction quadruplets from config file \"{}\"", configFile));
            }
        }
        else
            Log(Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Heisenberg: Using default configuration!");

        // Return
        Log(Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Heisenberg:");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {} {} {}", "boundary conditions", boundary_conditions[0], boundary_conditions[1], boundary_conditions[2]));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {}", "external field", B));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {}", "field_normal", B_normal.transpose()));
        if (anisotropy_from_file)
            Log(Log_Level::Parameter, Log_Sender::IO, "        K                     from file");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {}", "anisotropy[0]", K));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {}", "anisotropy_normal[0]", K_normal.transpose()));
        if (hamiltonian_type == "heisenberg_neighbours")
        {
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {}", "n_shells_exchange", n_shells_exchange));
            if (n_shells_exchange > 0)
                Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {}", "J_ij[0]", exchange_magnitudes[0]));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {}", "n_shells_dmi", n_shells_dmi));
            if (n_shells_dmi > 0)
                Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {}", "D_ij[0]", dmi_magnitudes[0]));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {}", "DM chirality", dm_chirality));
        }
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {}", "ddi_method", ddi_method_str));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = ({} {} {})", "ddi_n_periodic_images", ddi_n_periodic_images[0], ddi_n_periodic_images[1], ddi_n_periodic_images[2]));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {}", "ddi_radius", ddi_radius));

        std::unique_ptr<Engine::Hamiltonian_Heisenberg> hamiltonian;

        if (hamiltonian_type == "heisenberg_neighbours")
        {
            hamiltonian = std::unique_ptr<Engine::Hamiltonian_Heisenberg>(new Engine::Hamiltonian_Heisenberg(
                B, B_normal,
                anisotropy_index, anisotropy_magnitude, anisotropy_normal,
                exchange_magnitudes,
                dmi_magnitudes, dm_chirality,
                ddi_method, ddi_n_periodic_images, ddi_radius,
                quadruplets, quadruplet_magnitudes,
                geometry,
                boundary_conditions
            ));
        }
        else
        {
            hamiltonian = std::unique_ptr<Engine::Hamiltonian_Heisenberg>(new Engine::Hamiltonian_Heisenberg(
                B, B_normal,
                anisotropy_index, anisotropy_magnitude, anisotropy_normal,
                exchange_pairs, exchange_magnitudes,
                dmi_pairs, dmi_magnitudes, dmi_normals,
                ddi_method, ddi_n_periodic_images, ddi_radius,
                quadruplets, quadruplet_magnitudes,
                geometry,
                boundary_conditions
            ));
        }
        Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Heisenberg: built");
        return hamiltonian;
    }// end Hamiltonian_Heisenberg_From_Config


    std::unique_ptr<Engine::Hamiltonian_Micromagnetic> Hamiltonian_Micromagnetic_from_Config(const std::string configFile, const std::shared_ptr<Data::Geometry> geometry)
    {
        //-------------- Insert default values here -----------------------------
        // Boundary conditions (a, b, c)
        std::vector<int> boundary_conditions_i = { 0, 0, 0 };
        intfield boundary_conditions = { false, false, false };

        // The order of the finite difference approximation of the spatial gradient
        int spatial_gradient_order = 1;

        // External Magnetic Field
        int region_num;
        intfield regions = intfield(geometry->nos, 0);

        regionbook regions_book;
        VulkanCompute::VulkanSpiritLaunchConfiguration launchConfiguration;

        scalarfield Ms;
        scalarfield external_field_magnitude;
        vectorfield external_field_normal;
        Vector3 cell_sizes;
        scalar anisotropy_magnitude;
        intfield n_anisotropies;
        std::vector<std::vector<scalar>> anisotropy_magnitudes;
        std::vector<std::vector<Vector3>> anisotropy_normals;
        Vector3 anisotropy_normal;
        Matrix3 anisotropy_tensor;
        scalarfield  exchange_stiffness;
        Matrix3 exchange_tensor;
        scalarfield  dmi;
        scalarfield ddi;
        Matrix3 dmi_tensor;
        //------------------------------- Parser --------------------------------
        Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Micromagnetic: building");
        try
        {
            IO::Filter_File_Handle myfile(configFile);
          
            myfile.Read_Single(launchConfiguration.GPU_ID, "GPU_ID");
            myfile.Read_Single(launchConfiguration.savePeriod, "save_period");
            myfile.Read_Single(launchConfiguration.groupedIterations, "grouped_iterations");
            myfile.Read_Single(launchConfiguration.maxTorque, "max_torque");
            std::string double_precision_rotate;
            myfile.Read_String(double_precision_rotate, "double_precision_rotate");
            if (double_precision_rotate == "ON")
                launchConfiguration.double_precision_rotate = true;
            else
                launchConfiguration.double_precision_rotate = false;
            myfile.Read_String(double_precision_rotate, "double_precision_gradient");
            if (double_precision_rotate == "ON")
                launchConfiguration.double_precision_gradient = true;
            else
                launchConfiguration.double_precision_gradient = false;
            std::string save_gradient_components;
            myfile.Read_String(save_gradient_components, "save_gradient_components");
            if (save_gradient_components == "ON")
                launchConfiguration.saveGradientComponents = true;
            else
                launchConfiguration.saveGradientComponents = false;
            std::string lbfgs_linesearch;
            myfile.Read_String(lbfgs_linesearch, "lbfgs_linesearch");
            if (lbfgs_linesearch == "ON")
                launchConfiguration.LBFGS_linesearch = true;
            else
                launchConfiguration.LBFGS_linesearch = false;
            
            myfile.Read_Single(region_num, "number_regions");
            
            regions_book = regionbook(region_num);

            // Boundary conditions
            myfile.Read_3Vector(boundary_conditions_i, "boundary_conditions");
            boundary_conditions[0] = (boundary_conditions_i[0] != 0);
            boundary_conditions[1] = (boundary_conditions_i[1] != 0);
            boundary_conditions[2] = (boundary_conditions_i[2] != 0);

            if (boundary_conditions_i[0] == 0)
                launchConfiguration.performZeropadding[0] = true;
            else
                launchConfiguration.performZeropadding[0] = false;

            if (boundary_conditions_i[1] == 0)
                launchConfiguration.performZeropadding[1] = true;
            else
                launchConfiguration.performZeropadding[1] = false;

            if (boundary_conditions_i[2] == 0)
                launchConfiguration.performZeropadding[2] = true;
            else
                launchConfiguration.performZeropadding[2] = false;

            for (int i = 0; i < region_num; i++)
            {
                regions_book[i].periodic[0] = (launchConfiguration.performZeropadding[0]) ? 0 : 1;
                regions_book[i].periodic[1] = (launchConfiguration.performZeropadding[1]) ? 0 : 1;
                regions_book[i].periodic[2] = (launchConfiguration.performZeropadding[2]) ? 0 : 1;
                
            }
            if (myfile.Find("cell_sizes"))
            {
                myfile.iss >> cell_sizes[0] >> cell_sizes[1] >> cell_sizes[2];
                //cell_sizes[0] *= 5*sqrt(2 * 13e-12 / ((4 * 3.14159265358979) * 1e-7 * 800000 * 800000)) / 256;
                //cell_sizes[1] *= sqrt(2 * 13e-12 / ((4 * 3.14159265358979) * 1e-7 * 800000 * 800000)) / 64;
                //cell_sizes[2] *= 0.1*sqrt(2 * 13e-12 / ((4 * 3.14159265358979) * 1e-7 * 800000 * 800000)) / 1;
                for (int i = 0; i < region_num; i++)
                {
                   regions_book[i].cell_sizes = cell_sizes;
                   
                   regions_book[i].cell_sizes_inv[0] = 1.0 / cell_sizes[0];
                   regions_book[i].cell_sizes_inv[1] = 1.0 / cell_sizes[1];
                   regions_book[i].cell_sizes_inv[2] = 1.0 / cell_sizes[2];
                }
            }
            //Ms=scalarfield(region_num,0);
            if( myfile.Find("Ms") )
            {
            	for( int i = 0; i < region_num; i++ )
            	{
            		myfile.GetLine();
            		myfile.iss >> regions_book[i].Ms;
                    if (regions_book[i].Ms != 0)
                        regions_book[i].Ms_inv = 1.0 / regions_book[i].Ms;
                    else
                        regions_book[i].Ms_inv = 0;
				}
			}
            if (myfile.Find("alpha"))
            {
                launchConfiguration.damping = true;
                for (int i = 0; i < region_num; i++)
                {
                    myfile.GetLine();
                    myfile.iss >> regions_book[i].alpha;
                }
            }
            else {
                launchConfiguration.damping = false;
            }
            if (myfile.Find("frozen_spins"))
            {
                for (int i = 0; i < region_num; i++)
                {
                    myfile.GetLine();
                    myfile.iss >> regions_book[i].frozen_spins;
                }
            }
            else {
                for (int i = 0; i < region_num; i++)
                {
                    regions_book[i].frozen_spins=0;
                }
            }
            myfile.Read_Single(launchConfiguration.kernel_accuracy, "kernel_accuracy");
            // Precision of the spatial gradient calculation
            myfile.Read_Single(spatial_gradient_order, "spatial_gradient_order");
            myfile.Read_Single(launchConfiguration.max_move, "max_move");

            //launchConfiguration.max_move = 3.14159265358979 / launchConfiguration.max_move;
            myfile.Read_Single(launchConfiguration.n_lbfgs_memory, "n_lbfgs");
            // Field
            //external_field_magnitude=scalarfield(region_num,0);
			if( myfile.Find("external_field_magnitude") )
			{
				for( int i = 0; i < region_num; i++ )
				{
					myfile.GetLine();
					myfile.iss >> regions_book[i].external_field_magnitude;
				}
			}
			//external_field_normal=vectorfield(region_num, Vector3{0,0,1});
			if( myfile.Find("external_field_normal") )
			{
				for( int i = 0; i < region_num; i++ )
				{
					myfile.GetLine();
					myfile.iss >> regions_book[i].external_field_normal[0] >> regions_book[i].external_field_normal[1] >> regions_book[i].external_field_normal[2];
                    regions_book[i].external_field_normal.normalize();
					if (regions_book[i].external_field_normal.norm() < 1e-8)
					{
                        regions_book[i].external_field_normal = { 0,0,1 };
						Log(Log_Level::Warning, Log_Sender::IO, "Input for 'external_field_normal' had norm zero and has been set to (0,0,1)");
					}
				}
			}

            // TODO: anisotropy
			//n_anisotropies=intfield(region_num,0);
			if( myfile.Find("number_anisotropies") )
			{
				for( int i = 0; i < region_num; i++ )
				{
					myfile.GetLine();
					myfile.iss >> regions_book[i].n_anisotropies;
				}
			}
			//anisotropy_magnitudes=std::vector<std::vector<scalar>>(region_num);
			//anisotropy_normals=std::vector<std::vector<Vector3>>(region_num);
			/*for( int i = 0; i < region_num; i++ )
			{
                regions_book[i].anisotropy_magnitudes=std::vector<scalar>(n_anisotropies[i],0);
                regions_book[i].anisotropy_normals=std::vector<Vector3>(n_anisotropies[i],Vector3 {0,0,1});
			}*/

            if( myfile.Find("anisotropies_vectors") )
			{
				for( int i = 0; i < region_num; i++ )
				{
					for( int j = 0; j < regions_book[i].n_anisotropies; j++ )
					{
						myfile.GetLine();
						myfile.iss >> regions_book[i].anisotropy_normals[j][0] >> regions_book[i].anisotropy_normals[j][1] >> regions_book[i].anisotropy_normals[j][2] >> regions_book[i].anisotropy_magnitudes[j];
					}
				}
			}
             if( myfile.Find("cubic") )
			{
				for( int i = 0; i < region_num; i++ )
				{

						myfile.GetLine();
                        myfile.iss >> regions_book[i].anisotropy_cubic_normals[0] >> regions_book[i].anisotropy_cubic_normals[1] >> regions_book[i].anisotropy_cubic_normals[2] >> regions_book[i].anisotropy_cubic_normals[3] >> regions_book[i].anisotropy_cubic_normals[4] >> regions_book[i].anisotropy_cubic_normals[5] >> regions_book[i].anisotropy_cubic_normals[6] >> regions_book[i].anisotropy_cubic_normals[7] >> regions_book[i].anisotropy_cubic_normals[8] >> regions_book[i].Kc1;

				}
			}
            // TODO: exchange
            //exchange_stiffness=scalarfield(region_num,0);
            if( myfile.Find("exchange_stiffness") )
            {
            	for( int i = 0; i < region_num; i++ )
            	{
					myfile.GetLine();
					myfile.iss >> regions_book[i].Aexch;
                    if (regions_book[i].Aexch != 0)
                        regions_book[i].Aexch_inv = 1.0 / regions_book[i].Aexch;
                    else
                        regions_book[i].Aexch_inv = 0;
				}
            }
            //dmi=scalarfield(region_num,0);
			if( myfile.Find("dmi_interface") )
			{
				for( int i = 0; i < region_num; i++ )
				{
					myfile.GetLine();
					myfile.iss >> regions_book[i].Dmi_interface;
				}
			}
            if (myfile.Find("dmi_bulk"))
            {
                for (int i = 0; i < region_num; i++)
                {
                    myfile.GetLine();
                    myfile.iss >> regions_book[i].Dmi_bulk;
                }
            }
            std::string adaptive_time_step;
            myfile.Read_String(adaptive_time_step, "adaptive_time_step");
            if (adaptive_time_step == "ON")
                launchConfiguration.adaptiveTimeStep = true;
            else {
                launchConfiguration.adaptiveTimeStep = false;
            }
            scalar dt = 0;
            myfile.Read_Single(dt, "correct_dt");
            launchConfiguration.gamma = dt * 0.176085964411;
            //ddi = scalarfield(region_num, 0);
            if (myfile.Find("ddi"))
            {
                launchConfiguration.DDI = true;
                for (int i = 0; i < region_num; i++)
                {
                    myfile.GetLine();
                    myfile.iss >> regions_book[i].DDI;
                }
            }
            else {
                launchConfiguration.DDI = false;
            }
            
            for (int i = 0; i <  8*geometry->n_cells[0] * geometry->n_cells[1]; i++) {
                //regions[i] = 1;
            }
            for (int i = (int)(geometry->nos) - 8 * geometry->n_cells[0] * geometry->n_cells[1]; i < ((int)(geometry->nos)); i++) {
                //regions[i] = 1;
            }

            for (int i = 0; i < ((int)(geometry->nos)); i++) {
                int z = i / (geometry->n_cells[0] * geometry->n_cells[1]);
                int y = (i - z * geometry->n_cells[0] * geometry->n_cells[1]) / geometry->n_cells[0];
                int x = i - z * geometry->n_cells[0] * geometry->n_cells[1] - y * geometry->n_cells[0];
                if (pow(((float)(x - geometry->n_cells[0] / 2) + 0.5), 2.0) + pow(((float)(y - geometry->n_cells[1] / 2) + 0.5), 2.0) >= 0.25f * geometry->n_cells[0] * geometry->n_cells[0])
                {
                    //std::cout << x << " " << y << " " << z << "\n";
                    //regions[i] = 2;
                 
                }
            }
           // std::cout << nn <<"\n";
           /* std::string ddi_bool;
            myfile.Read_String(ddi_bool, "ddi");
            if (ddi_bool == "ON")
                launchConfiguration.DDI = true;
            else
                launchConfiguration.DDI = false;*/
            // TODO: dipolar
            Log(Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Micromagnetic:");
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {} {} {}", "boundary conditions", boundary_conditions[0], boundary_conditions[1], boundary_conditions[2]));
            /*Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {}", "external field", field));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {}", "field_normal", field_normal.transpose()));
            for (int i=0;i<n_anisotropies;i++){
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {} {}", "anisotropy_magnitude", i, anisotropy_magnitudes[i]));
            }
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {:<21} = {}", "exchange_magnitude", exchange_tensor(0,0)));*/

            auto hamiltonian = std::unique_ptr<Engine::Hamiltonian_Micromagnetic>(new Engine::Hamiltonian_Micromagnetic(
                boundary_conditions,
                geometry,
                region_num,
                regions,
                regions_book,
                launchConfiguration
            ));
            Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Micromagnetic: built");
            return hamiltonian;
        }// end try
        catch( ... )
        {
            spirit_handle_exception_core(fmt::format(
                "Unable to parse all parameters of the Micromagnetic Hamiltonian from \"{}\"", configFile));
            return NULL;
        }
		// Return
		

    }// end Hamiltonian_Micromagnetic_from_Config


    std::unique_ptr<Engine::Hamiltonian_Gaussian> Hamiltonian_Gaussian_from_Config(const std::string configFile, std::shared_ptr<Data::Geometry> geometry)
    {
        //-------------- Insert default values here -----------------------------
        // Number of Gaussians
        int n_gaussians = 1;
        // Amplitudes
        std::vector<scalar> amplitude = { 1 };
        // Widths
        std::vector<scalar> width = { 1 };
        // Centers
        std::vector<Vector3> center = { Vector3{ 0, 0, 1 } };

        //------------------------------- Parser --------------------------------
        Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Gaussian: building");

        if( configFile != "" )
        {
            try
            {
                IO::Filter_File_Handle myfile(configFile);

                // N
                myfile.Read_Single(n_gaussians, "n_gaussians");

                // Allocate arrays
                amplitude = std::vector<scalar>(n_gaussians, 1.0);
                width = std::vector<scalar>(n_gaussians, 1.0);
                center = std::vector<Vector3>(n_gaussians, Vector3{0, 0, 1});
                // Read arrays
                if (myfile.Find("gaussians"))
                {
                    for (int i = 0; i < n_gaussians; ++i)
                    {
                        myfile.GetLine();
                        myfile.iss >> amplitude[i];
                        myfile.iss >> width[i];
                        for (int j = 0; j < 3; ++j)
                        {
                            myfile.iss >> center[i][j];
                        }
                        center[i].normalize();
                    }
                }
                else Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Gaussian: Keyword 'gaussians' not found. Using Default: {0, 0, 1}");
            }// end try
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Unable to read Hamiltonian_Gaussian parameters from config file \"{}\"", configFile));
            }
        }
        else
            Log(Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Gaussian: Using default configuration!");


        // Return
        Log(Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Gaussian:");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<12} = {1}", "n_gaussians", n_gaussians));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<12} = {1}", "amplitude[0]", amplitude[0]));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<12} = {1}", "width[0]", width[0]));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<12} = {1}", "center[0]", center[0].transpose()));
        auto hamiltonian = std::unique_ptr<Engine::Hamiltonian_Gaussian>(new Engine::Hamiltonian_Gaussian(
            amplitude, width, center
        ));
        Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Gaussian: built");
        return hamiltonian;
    }
}// end namespace IO
