#pragma once
#ifndef SIMPLE_FMM_BOX_HPP
#define SIMPLE_FMM_BOX_HPP
#include "fmm/SimpleFMM_Defines.hpp"
#include <vector>
#include <map>

//TODO add iterator over contained positions
namespace SimpleFMM 
{
    enum BoxType 
    {
        //R = right, L = left, F = Front, B = Back, U = UP, B = Down
        //Faces
        R_Face,
        L_Face,
        U_Face,
        D_Face,
        F_Face,
        B_Face,
        //Edges
        RF_Edge,
        RB_Edge,
        RU_Edge,
        RD_Edge,
        LF_Edge,
        LB_Edge,
        LU_Edge,
        LD_Edge,
        DF_Edge,
        DB_Edge,
        UF_Edge,
        UB_Edge,
        //Corners
        FRD_Corner,
        FRU_Corner,
        FLD_Corner,
        FLU_Corner,
        BRD_Corner,
        BRU_Corner,
        BLD_Corner,
        BLU_Corner,
        // Center
        Bulk,
        // Default (i.e did not care to specify)
        Default
    };

    struct Box
    {
        // === DATA ===
        // Tree Structure
        int id         = -1;
        int level      = -1;
        int n_children = -1;

        // Information about the geometry of the box
        const vectorfield & pos; //refers to a Vector3 of positions
        intfield pos_indices; //the positions at these indices are 'contained' in the box
        int n_spins;
        Vector3 center;

        scalar radius;
        Vector3 min;
        Vector3 max;

        //caches
        std::map<int, complexfield> M2M_cache;
        // std::map<int, complexfield> M2L_cache;
        std::map<int, MatrixXc>     M2L_cache;
        std::map<int, complexfield> L2L_cache;
        complexfield Farfield_cache;

        // Multipole Expansion
        intfield interaction_list; //the ids of the boxes with which this box interacts via multipoles
        int l_min = 2;
        int l_max;
        std::vector<Matrix3c> multipole_hessians;
        std::vector<Vector3c> multipole_moments;
        std::vector<Vector3c> local_moments;

        // === METHODS ===
        // Constructs a box that contains all the positions in the vectorfield and calculates its boundaries
        Box(const vectorfield & pos, int level, int l_max=6);
        // Constructs a box that contains all the positions that are contained in 'indices' and calculates the boundaries
        Box(const vectorfield & pos, intfield indices, int level, int l_max=6);
        // Constructs a box that contains all the positions that are contained in 'indices' and takes the bondaries as arguments
        Box(const vectorfield & pos, intfield indices, int level, Vector3 min, Vector3 max, int l_max=6);
        void Update(int l_max);

        std::vector<Box> Divide_Evenly(int n_dim = 3, Vector3 normal1 = {0,0,1}, Vector3 normal2 = {0,1,0});
        void Get_Covering_Circle();
        void Get_Boundaries();
        bool Is_Near_Neighbour(Box& other_box);
        bool Fulfills_MAC(Box& other_box);

        void Clear_Moments();

        // These are functions that are mainly used for debugging
        void Evaluate_Near_Field(const vectorfield& spins, const scalarfield& mu_s, vectorfield& gradient);
        // void Evaluate_Far_Field(vectorfield& gradient);
        // void Build_Far_Field_Cache();
        Vector3  Evaluate_Directly_At(Vector3 r, vectorfield& spins);
        Vector3c Evaluate_Multipole_Expansion_At(Vector3 r);
        Vector3  Evaluate_Far_Field_At(Vector3 r);
        void Print_Info(bool print_multipole_moments = false, bool print_local_moments = false);

    };
}

#endif