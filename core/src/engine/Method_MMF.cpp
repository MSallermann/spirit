#include "Method_MMF.h"

namespace Engine
{
    Method_MMF::Method_MMF(std::shared_ptr<Data::Parameters_MMF> parameters, int idx_img, int idx_chain) :
        Method(parameters, idx_img, idx_chain)
    {

    }

    void Method_MMF::Calculate_Force(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces)
    {

    }
		
    // Check if the Forces are converged
    bool Method_MMF::Force_Converged()
    {
        return false;
    }

    // Optimizer name as string
    std::string Method_MMF::Name() { return "MMF"; }
}