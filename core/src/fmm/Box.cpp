#include "fmm/Box.hpp"
#include "fmm/Spherical_Harmonics.hpp"
#include "fmm/Utility.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>
namespace SimpleFMM 
{
    using Utility::multipole_idx;
    using Utility::n_moments;
    Box::Box(const vectorfield & pos) : 
    pos(pos)
    { }

    Box::Box(const vectorfield & pos, intfield indices, int level, int n_bands):
    level(level), pos(pos), pos_indices(indices), n_bands(n_bands)
    {
        Get_Boundaries();
        Get_Covering_Circle();
        Update(n_bands);
    }

    void Box::Update(int n_bands)
    {
        this->multipole_moments.resize(n_moments(n_bands-1, l_min));
        this->multipole_moments.shrink_to_fit();
        this->local_moments.resize(n_moments(n_bands - 1));
        this->local_moments.shrink_to_fit();
    }

    //Divides a Box evenly by cutting it in two in every dimension
    //TODO: Use normal1 and normal2 ...
    std::vector<Box> Box::Divide_Evenly(int n_dim, Vector3 normal1, Vector3 normal2)
    {
        // std::vector<Box> result(8);
        int n_children = std::pow(2, n_dim);

        if(this->pos_indices.size() < n_children)
        {
            throw std::invalid_argument("This box can't be further divided! At least one of the children would be empty.");
        }

        std::vector<intfield> result_indices(n_children);
        this->n_children = std::pow(2, n_dim);

        for(int idx = 0; idx < pos_indices.size(); idx++)
        {
            auto& i = pos_indices[idx];
            intfield tupel(n_dim);
            intfield maxVal(n_dim, 2);
            for(int dim=0; dim<n_dim; dim++)
            {
                tupel[dim] = (pos[i][dim] < center_of_mass[dim]) ? 0 : 1;
            }
            result_indices[Utility::idx_from_tupel(tupel, maxVal)].push_back(i);
        }

        std::vector<Box> result;
        for(int i = 0; i < n_children; i++)
        {
            result.emplace_back(pos, result_indices[i], level+1, this->n_bands);
        }       
        return result;
    }
    
    void Box::Get_Covering_Circle()
    {
        center_of_mass = {0, 0, 0};
        for(auto i : pos_indices)
            center_of_mass += pos[i]/pos_indices.size();

        radius = 0;
        for(auto i : pos_indices)
        {
            scalar curr = (this->center_of_mass - pos[i]).norm();
            if(curr > radius)
                radius = curr;
            curr = 0;
        }
    }

    //Computes the extents of a box 
    void Box::Get_Boundaries()
    {
        middle = {0, 0, 0};

        min = pos[pos_indices[0]];
        max = pos[pos_indices[0]];
        
        for(int idx = 0; idx < pos_indices.size(); idx++)
        {
            auto& i = pos_indices[idx];
            for(int dir = 0; dir < 3; dir++)
            {
                if(min[dir] > pos[i][dir])
                    min[dir] = pos[i][dir];
                if(max[dir] < pos[i][dir])
                    max[dir] = pos[i][dir];
            }
        }
        middle = 0.5 * (max + min);
    }  

    //Check if other_box is a near neighbour of this box
    bool Box::Is_Near_Neighbour(Box& other_box)
    {
        scalar distance = (this->center_of_mass - other_box.center_of_mass).norm();
        return distance < 4 * std::max(radius, other_box.radius);
    }

    //Test if the MAC is fulfilled between two boxes
    bool Box::Fulfills_MAC(Box& other_box)
    {
        bool on_the_same_level = (this->level == other_box.level);
        return on_the_same_level && !Is_Near_Neighbour(other_box);
    }

    //Get the hessian of the estatic multipole moments via finite difference
    void Box::Get_Multipole_Hessians(scalar epsilon)
    {
        for(int l = l_min; l < this->n_bands; l++)
        {
            for(int m = -l; m <= l; m++)
            {
                for(auto p_idx : pos_indices)
                {   
                    //Calculate the hessian via finite difference
                    Matrix3c hessian;
                    vectorfield d_xyz = {
                                            {0.5 * epsilon, 0, 0},
                                            {0, 0.5 * epsilon, 0},
                                            {0, 0, 0.5 * epsilon}
                                        };

                    Vector3 spherical;
                     //fill the hessian with second derivatives
                    for(int dir1 = 0; dir1 < 3; dir1++)
                    {
                        for(int dir2 = dir1; dir2 < 3; dir2++)
                        {
                            Utility::get_spherical(pos[p_idx] + d_xyz[dir1] + d_xyz[dir2] - center_of_mass, spherical);
                            auto fpp = std::conj(Spherical_Harmonics::R(l, m, spherical[0], spherical[1], spherical[2]));

                            Utility::get_spherical(pos[p_idx] + d_xyz[dir1] - d_xyz[dir2] - center_of_mass, spherical);
                            auto fpm = std::conj(Spherical_Harmonics::R(l, m, spherical[0], spherical[1], spherical[2]));

                            Utility::get_spherical(pos[p_idx] - d_xyz[dir1] + d_xyz[dir2] - center_of_mass, spherical);
                            auto fmp = std::conj(Spherical_Harmonics::R(l, m, spherical[0], spherical[1], spherical[2]));

                            Utility::get_spherical(pos[p_idx] - d_xyz[dir1] - d_xyz[dir2] - center_of_mass, spherical);
                            auto fmm = std::conj(Spherical_Harmonics::R(l, m, spherical[0], spherical[1], spherical[2]));

                            hessian(dir1, dir2) = 1/(epsilon * epsilon) * (fpp - fpm - fmp + fmm);
                            if(dir1 != dir2)
                                hessian(dir2, dir1) = hessian(dir1, dir2);
                        }
                    }
                    this->multipole_hessians.push_back(hessian);
                }
            }
        }
    }

    // Calculate the multipole moments of a box from the spins contained in it
    void Box::Calculate_Multipole_Moments(const vectorfield& spins, const scalarfield& mu_s)
    {
        for(int i = 0; i<pos_indices.size(); i++)
        {
            auto p_idx = pos_indices[i];
            for(int l = l_min; l < n_bands; l++)
            {
                for(int m = -l; m <= l; m++)
                {
                    this->multipole_moments[Utility::multipole_idx(l, m, l_min)] += this->multipole_hessians[Utility::multipole_idx(l, m, l_min) * pos_indices.size() + i] * spins[p_idx] * mu_s[p_idx];
                }
            }
        }
    }

    //Translates the multipole moments of child_box and adds them to this box
    //TODO: This is an O(p^4) method, but more efficient O(p^3) methods are available
    void Box::Add_Multipole_Moments(Box& child_box)
    {
        auto diff = child_box.center_of_mass - this->center_of_mass;
        Vector3 diff_sph;
        Utility::get_spherical(diff, diff_sph);
        for(int l = l_min; l < n_bands; l++)
        {
            for(int m = -l; m <= l; m++)
            {
                for(int lc = l_min; lc <= l; lc++)
                {
                    for(int mc = std::max(-lc, m+lc-l); mc <= std::min(lc, m+l-lc); mc++) //because we need |m-mc| <= l-lc
                    {
                        this->multipole_moments[Utility::multipole_idx(l, m, l_min)] += child_box.multipole_moments[Utility::multipole_idx(lc, mc, l_min)] * std::conj(Spherical_Harmonics::R(l-lc, m-mc, diff_sph[0], diff_sph[1], diff_sph[2]));
                    }
                }
            }
        }
    }

    std::complex<scalar> _M2L_prefactor(Vector3 diff_sph, int l, int lp, int m, int mp)
    {
        return std::pow(-1,lp) * Spherical_Harmonics::S(l+lp, m+mp, diff_sph[0], diff_sph[1], diff_sph[2]);
    }

    void Box::M2L(Box& other_box)
    {
        Vector3 diff_sph;
        Utility::get_spherical(this->center_of_mass - other_box.center_of_mass, diff_sph);
    
        for(int l = l_min; l < n_bands; l++)
        {
            for(int m = -l; m < l+1; m++)
            {
                for(int lp = 0; lp < n_bands; lp++)
                {
                    for(int mp = -lp; mp < lp + 1; mp++)
                    {
                        this->local_moments[Utility::multipole_idx(lp, mp)] += ((_M2L_prefactor(diff_sph, l, lp, m, mp) * other_box.multipole_moments[Utility::multipole_idx(l, m, l_min)])).conjugate();
                    }
                }
            }
        }
    }

    void Box::Add_Local_Moments(Box& child_box)
    {
        Vector3 diff_sph;
        Utility::get_spherical(this->center_of_mass - child_box.center_of_mass, diff_sph);
        for(int l = 0; l < n_bands; l++)
        {
            for(int m = -l; m <= l; m++)
            {
                for(int lp = l; lp < n_bands; lp++)
                {
                    for(int mp = std::max(-lp, m-lp+l); mp <= std::min(lp+1, m+lp-l); mp++)
                    {
                        child_box.local_moments[Utility::multipole_idx(l, m)] += this->local_moments[Utility::multipole_idx(lp, mp)] * std::conj(Spherical_Harmonics::R(lp-l, mp-m, diff_sph[0], diff_sph[1], diff_sph[2]));
                    }
                }
            }
        }
    }

    void Box::Evaluate_Near_Field(const vectorfield& spins, vectorfield& gradient)
    {
        //TODO check if this is working correctly
        for(int i = 0; i < pos_indices.size(); i++)
        {
            for(int j = i+1; j < pos_indices.size(); j++)
            {
                auto& idx1 = pos_indices[i];
                auto& idx2 = pos_indices[j];
                auto r12 = pos[idx1] - pos[idx2];
                auto r = r12.norm();
                gradient[idx1] += 3 * spins[idx2].dot(r12) * r12 / std::pow(r, 5) - spins[idx2] / std::pow(r,3);
                gradient[idx2] += 3 * spins[idx1].dot(r12) * r12 / std::pow(r, 5) - spins[idx1] / std::pow(r,3);
            }
        }
    }

    void Box::Interact_Directly(Box& box, const vectorfield& spins, vectorfield& gradient)
    {
        for(int i = 0; i < this->pos_indices.size(); i++)
        {
            for(int j = 0; j < box.pos_indices.size(); j++)
            {
                auto& idx1 = this->pos_indices[i];
                auto& idx2 = box.pos_indices[j];
                auto r12 = pos[idx1] - pos[idx2];
                auto r = r12.norm();
                gradient[idx1] += 3 * spins[idx2].dot(r12) * r12 / std::pow(r, 5) - spins[idx2] / std::pow(r,3);
            }
        }
    }

    void Box::Evaluate_Far_Field(vectorfield& gradient)
    {
        for(auto p_idx : pos_indices)
        {               
            Vector3 p_sph;
            Utility::get_spherical(pos[p_idx] - this->center_of_mass, p_sph);

            for(int l=0; l<n_bands; l++)
            {
                for(int m=-l; m<=l; m++)
                {
                    auto& moment = this->local_moments[Utility::multipole_idx(l,m)];
                    gradient[p_idx] += (moment * Spherical_Harmonics::R(l, m, p_sph[0], p_sph[1], p_sph[2])).real();
                }
            }
        }
    }

    Vector3 Box::Evaluate_Directly_At(Vector3 r, vectorfield& spins)
    {
        Vector3 result;
        for(auto p_idx : pos_indices)
        {
            auto r12 = pos[p_idx] - r;
            auto r = r12.norm();
            result += 3 * spins[p_idx].dot(r12) * r12 / std::pow(r, 5) - spins[p_idx] / std::pow(r,3);
        }
        return result;
    }

    Vector3 Box::Evaluate_Far_Field_At(Vector3 r)
    {
        Vector3 p_sph;
        Utility::get_spherical(r - this->center_of_mass, p_sph);
        Vector3 temp;
        for(int l=0; l<n_bands; l++)
        {
            for(int m=-l; m<=l; m++)
            {
                auto& moment = this->local_moments[Utility::multipole_idx(l,m)];        
                temp += (moment * Spherical_Harmonics::R(l, m, p_sph[0], p_sph[1], p_sph[2])).real();
            }
        }
        return temp;
    }

    Vector3c Box::Evaluate_Multipole_Expansion_At(Vector3 r)
    {
        Vector3c result;
        Vector3 r_sph;
        Utility::get_spherical(r - this->center_of_mass, r_sph);
        for(auto l=l_min; l<n_bands; l++)
        {
            for(auto m=-l; m<=l; m++)
            {
                result += this->multipole_moments[multipole_idx(l, m, l_min)] * Spherical_Harmonics::S(l, m, r_sph[0], r_sph[1], r_sph[2]);
            }
        }
        return result;
    }

    //Mainly for debugging
    void Box::Print_Info(bool print_multipole_moments, bool print_local_moments)
    {
        std::cout   << "-------------- Box Info --------------" << std::endl
                    << "ID = "<< id <<", Level = " << level << std::endl
                    << "n_children = " << this->n_children << std::endl
                    << "Number of Particles = " 
                    << pos_indices.size()
                    << std::endl
                    << "Middle              = " 
                    << middle[0] << " "
                    << middle[1] << " "
                    << middle[2] << " " << std::endl
                    << "Center_of_Mass      = " 
                    << center_of_mass[0] << " "
                    << center_of_mass[1] << " "
                    << center_of_mass[2] << " " << std::endl
                    << "Radius = " << radius << std::endl
                    << "Min / Max "             << std::endl
                    << "  x: " << min[0] << " / " << max[0] << std::endl
                    << "  y: " << min[1] << " / " << max[1] << std::endl
                    << "  z: " << min[2] << " / " << max[2] << std::endl
                    << "Number of Interaction Boxes = " << interaction_list.size() << std::endl;
                    if(print_multipole_moments)
                    {
                        std::cout << "== Multipole Moments == " << std::endl;
                        for(auto l=l_min; l<n_bands; l++)
                        {
                            for(auto m=-l; m<=l; m++) 
                            {       
                                std::cout << ">> --- l = "<< l << ", m = " << m << " -- <<" << std::endl;
                                std::cout << multipole_moments[Utility::multipole_idx(l, m, l_min)] << std::endl;
                            }
                        }
                    }
                    if(print_local_moments)
                    {
                        std::cout << "== Local Moments == " << std::endl;
                        for(auto l=0; l<n_bands; l++)
                        {
                            for(auto m=-l; m<=l; m++) 
                            {       
                                std::cout << ">> --- l = "<< l << ", m = " << m << " -- <<" << std::endl;
                                std::cout << local_moments[Utility::multipole_idx(l, m)] << std::endl;
                            }
                        }
                    }
                    std::cout << "Interaction List: " << std::endl;
                    for(auto i : interaction_list)
                        std::cout << i << " ";
                    std::cout << "\nNear Neighbours: " << std::endl;
                    for(auto i : near_neighbour_indices)
                        std::cout << i << " ";
        std::cout << std::endl;                
    }

    
}
