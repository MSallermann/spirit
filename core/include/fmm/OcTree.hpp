#pragma once
#ifndef SIMPLE_FMM_OCTREE_HPP 
#define SIMPLE_FMM_OCTREE_HPP

#include "SimpleFMM_Defines.hpp"
#include "Box.hpp"
#include <vector>
#include <iterator>

namespace SimpleFMM
{
    class OcTree
    {
        //TODO
        // 1. add functions that return **iterators** over
        //  levels
        //  children
        //  near_neighbours

        using iterator = Box*;

        std::vector<Box> boxes;
        intfield start_idx_level;
        intfield n_boxes_on_level;
        int _Get_Parent_Idx(int idx);
        int _Get_Child_Idx(int idx);

        public:
        int div;
        int n_level;
        int children_per_box;
        int n_boxes;
        int n_bands;

        OcTree();
        OcTree(int depth, vectorfield& pos, int n_dim = 3, int n_bands = 6);

        //Implement all these
        iterator begin_level(int level) 
        {
            return iterator(&boxes[start_idx_level[level]]);
        };

        iterator end_level(int level)
        {
            return iterator(&boxes[start_idx_level[level] + n_boxes_on_level[level]]);
        };

        iterator begin_children(Box& box)
        {
            return iterator(&boxes[_Get_Child_Idx(box.id)]);
        };

        iterator end_children(Box& box)
        {
            return iterator(&boxes[_Get_Child_Idx(box.id) + box.n_children]);
        };

        //TODO
        iterator begin_near_neighbours(Box& box);
        iterator end_near_neighbours(Box& box);

        Box& Get_Box(int idx);
        Box& Get_Parent(Box& box);

        void Upward_Pass(vectorfield& spins, scalarfield& mu_s);
        void Downward_Pass();
        void Evaluation(const vectorfield& spins, vectorfield& gradient);
        void Direct_Evaluation(const vectorfield& spins, vectorfield& gradient);
    };
    
}


#endif

