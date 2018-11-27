#pragma once
#ifndef SIMPLE_FMM_BOX_HPP
#define SIMPLE_FMM_BOX_HPP
#include "SimpleFMM_Defines.hpp"
#include <vector>

//TODO add iterator over contained positions
namespace SimpleFMM 
{
    struct Box
    {
        //=== DATA ===
        //Tree Structure
        int id         = -1;
        int level      = -1;
        int n_children = -1;
        intfield near_neighbour_indices;

        //Information about the geometry of the box
        const vectorfield & pos; //refers to a Vector3 of positions
        intfield pos_indices; //the positions at these indices are 'contained' in the box
        Vector3 center_of_mass;
        Vector3 middle;
        scalar radius;
        Vector3 min;
        Vector3 max;

        //Multipole Expansion
        intfield interaction_list; //the ids of the boxes with which this box interacts via multipoles
        int l_min = 2;
        int n_bands;
        std::vector<Matrix3c> multipole_hessians;
        std::vector<Vector3c> multipole_moments;
        std::vector<Vector3c> local_moments;

        //=== METHODS ===
        Box(const vectorfield & pos);
        Box(const vectorfield & pos, intfield indices, int level, int n_bands=6);
        void Update(int n_bands);

        std::vector<Box> Divide_Evenly(int n_dim = 3, Vector3 normal1 = {0,0,1}, Vector3 normal2 = {0,1,0});
        void Get_Covering_Circle();
        void Get_Boundaries();
        bool Is_Near_Neighbour(Box& other_box);
        bool Fulfills_MAC(Box& other_box);

        //TODO
        void Calculate_Multipole_Moments(const vectorfield& spins, const scalarfield& mu_s);
        void Add_Multipole_Moments(Box & child_box);

        //Convert the multipole moments of other_box into local moments about this boxes center and add them to the already existing local moments
        void M2L(Box& other_box);
        void Add_Local_Moments(Box& child_box);

        void Get_Multipole_Moment(int l, int m);
        void Get_Local_Expansion(Box& other_box);

        void Get_Multipole_Hessians(scalar epsilon = 1e-3);

        void Interact_Directly(Box& box, const vectorfield& spins, vectorfield& gradient);

        void Evaluate_Near_Field(const vectorfield& spins, vectorfield& gradient);
        void Evaluate_Far_Field(vectorfield& gradient);
        
        Vector3  Evaluate_Directly_At(Vector3 r, vectorfield& spins);
        Vector3c Evaluate_Multipole_Expansion_At(Vector3 r);
        Vector3  Evaluate_Far_Field_At(Vector3 r);

        void Print_Info(bool print_multipole_moments = false, bool print_local_moments = false);

    };
}

#endif