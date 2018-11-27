#include "fmm/OcTree.hpp"
#include "fmm/Utility.hpp"
#include <iostream>

#include <cmath>

namespace SimpleFMM
{
    OcTree::OcTree() {};

    //The OcTree constructor does the following
    //  1. Create a tree with n_level levels of boxes
    //  2. Each box on level l gets divided into 8 equally sized boxes from which level l+1 is formed
    //  3. Determine the interaction lists according to the MAC
    //  4. The boxes on the deepest level calculate the hessian of the multipole moments

    OcTree::OcTree(int n_level, vectorfield& pos, int n_dim, int n_bands)
    {
        this->n_boxes = 0;
        this->n_level = n_level;
        this->n_bands = n_bands;
        this->children_per_box = std::pow(2, n_dim);
        //could be replaced by formula for geometric sum
        for(int i = 0; i < n_level; i++)
        {
            n_boxes += std::pow(children_per_box, i);
        }

        //This is ** very ** important to avoid iterator invalidation on boxes.push_back(..)
        boxes.reserve(n_boxes);

        auto indices = SimpleFMM::intfield(pos.size());
        for(int i = 0; i < pos.size(); i++)
            indices[i] = i;
        
        //push back the root box
        this->boxes.push_back(Box(pos, indices, 0, n_bands));
        start_idx_level.push_back(0);
        n_boxes_on_level.push_back(1);
        Get_Box(0).id = 0;

        for(int level = 1; level < n_level; level++)
        {
            start_idx_level.push_back(boxes.size());
            //push back the children of all the boxes on the previous level 
            for(auto it = begin_level(level-1); it != end_level(level-1); it++)
            {
                for(auto box : it->Divide_Evenly(n_dim))
                {
                    box.id = boxes.size();
                    boxes.push_back(box);
                }
            }
            n_boxes_on_level.push_back(boxes.size() - start_idx_level[level]);
            
            //build the interaction lists
            //TODO: more efficient implementation

            //iterate over the parent level
            for(auto it_par = begin_level(level-1); it_par != end_level(level-1); it_par++)
            {
                //find boxes at the parent level which are near neighbours
                for(auto it_par_2 = begin_level(level-1); it_par_2 != end_level(level-1); it_par_2++)
                {
                    if(it_par->Is_Near_Neighbour(*it_par_2))
                    {
                        //if the children fulfill the mac at this level add them to the interactions list
                        for(auto it_ch = begin_children(*it_par); it_ch != end_children(*it_par); it_ch++)
                        {
                            for(auto it_ch_2 = begin_children(*it_par_2); it_ch_2 != end_children(*it_par_2); it_ch_2++)
                            {
                                if(it_ch->Fulfills_MAC(*it_ch_2))
                                {
                                    it_ch->interaction_list.push_back(it_ch_2->id);
                                }
                            }
                        }
                    }
                }
            }
        }

        //The boxes on the last level calculate the hessian of their estatic multipole moments and find the indices of their near neighbours
        for(auto it_last = begin_level(n_level-1); it_last != end_level(n_level-1); it_last++)
        {
            it_last->Get_Multipole_Hessians(1e-1);
            for (auto it_last2 = begin_level(n_level-1); it_last2 != end_level(n_level-1); it_last2++)
            {
                if(it_last->Is_Near_Neighbour(*it_last2))
                {
                    it_last->near_neighbour_indices.push_back(it_last2->id);
                }
            }
        }
    }

    void OcTree::Upward_Pass(vectorfield& spins, scalarfield& mu_s)
    {
        for(auto box = this->begin_level(n_level-1); box != this->end_level(n_level-1); box++)
        {
            box->Calculate_Multipole_Moments(spins, mu_s);
        }
        for(int lvl = n_level-2; lvl>=0; lvl--)
        {
            for(auto box = this->begin_level(lvl); box != this->end_level(lvl); box++)
            {
                for(auto c = begin_children(*box); c != end_children(*box); c++)
                {
                    box->Add_Multipole_Moments(*c);
                }
            }
        }
    }

    //Performs a naive O(N^2) summation for the gradient
    void OcTree::Direct_Evaluation(const vectorfield& spins, vectorfield& gradient)
    {
        boxes[0].Evaluate_Near_Field(spins, gradient);
    }

    void OcTree::Downward_Pass()
    {
        //From the coarset level to the finest
        for(int lvl = 0; lvl < n_level; lvl++)
        {
            //Each box calculates its local expansion due to the multipole expansions of the boxes in its interaction list
            for(auto box = begin_level(lvl); box != end_level(lvl); box++)
            {
                for(auto interaction_id : box->interaction_list)
                {
                    box->M2L(boxes[interaction_id]);
                }

                //Then the local expansions get translated down to the children
                if (lvl != n_level - 1)
                    for(auto child=begin_children(*box); child != end_children(*box); child++)
                        box->Add_Local_Moments(*child);
            }
        }
    }

    void OcTree::Evaluation(const vectorfield& spins, vectorfield& gradient)
    {
        for(auto leaf_box = begin_level(n_level-1); leaf_box != end_level(n_level-1); leaf_box++)
        {
            leaf_box->Evaluate_Far_Field(gradient);
            leaf_box->Evaluate_Near_Field(spins, gradient);
            for(int n_idx : leaf_box->near_neighbour_indices)
            {
                if(n_idx != leaf_box->id)
                    leaf_box->Interact_Directly(this->Get_Box(n_idx), spins, gradient);
            }
        }
    }

    int OcTree::_Get_Parent_Idx(int idx)
    {
        int dist_to_start = idx - start_idx_level[boxes[idx].level];
        return dist_to_start/children_per_box + start_idx_level[boxes[idx].level - 1];
    }

    int OcTree::_Get_Child_Idx(int idx)
    {
        int dist_to_start = idx - start_idx_level[boxes[idx].level];
        return start_idx_level[boxes[idx].level + 1] + children_per_box * dist_to_start;
    }

    Box& OcTree::Get_Box(int idx)
    {
        return this->boxes[idx];
    }

    Box& OcTree::Get_Parent(Box& box)
    {
        return this->boxes[_Get_Parent_Idx(box.id)];
    }

}