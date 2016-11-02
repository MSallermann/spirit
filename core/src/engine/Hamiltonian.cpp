#include <engine/Hamiltonian.h>

#include <utility/Vectormath.h>

namespace Engine
{
	Hamiltonian::Hamiltonian(std::vector<bool> boundary_conditions) :
        boundary_conditions(boundary_conditions)
    {
        prng = std::mt19937(94199188);
		distribution_int = std::uniform_int_distribution<int>(0, 1);

        delta = 1e-6;
    }

    void Hamiltonian::Hessian(const std::vector<double> & spins, std::vector<double> & hessian)
    {
		// This is a regular finite difference implementation (probably not very efficient)
		// see https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm

		int nos = spins.size() / 3;

		// Calculate finite difference
		std::vector<double> spins_pp(3 * nos, 0);
		std::vector<double> spins_mm(3 * nos, 0);
		std::vector<double> spins_pm(3 * nos, 0);
		std::vector<double> spins_mp(3 * nos, 0);

		for (int i = 0; i < 3 * nos; ++i)
		{
			for (int j = 0; j < 3 * nos; ++j)
			{
				if (i == j)
				{
					spins_pp = spins;
					spins_mm = spins;
					spins_pm = spins;
					spins_mp = spins;

					spins_pp[i] = spins_pp[i] + 2.0*delta;
					spins_mm[i] = spins_mm[i] - 2.0*delta;
					spins_pm[i] = spins_mm[i] + delta;
					spins_mp[i] = spins_mm[i] - delta;

					Utility::Vectormath::Normalize_3Nos(spins_pp);
					Utility::Vectormath::Normalize_3Nos(spins_mm);
					Utility::Vectormath::Normalize_3Nos(spins_pm);
					Utility::Vectormath::Normalize_3Nos(spins_mp);

					double E_pp = this->Energy(spins_pp);
					double E_mm = this->Energy(spins_mm);
					double E_pm = this->Energy(spins_pm);
					double E_mp = this->Energy(spins_mp);
					double E = this->Energy(spins);

					hessian[i * 3 * nos + j] = (-E_pp +16*E_pm - 30*E + 16*E_mp - E_mm) / (12 * delta*delta);
				}
				else
				{
					spins_pp = spins;
					spins_mm = spins;
					spins_pm = spins;
					spins_mp = spins;

					spins_pp[i] = spins_pp[i] + delta;
					spins_pp[j] = spins_pp[j] + delta;
					spins_mm[i] = spins_mm[i] - delta;
					spins_mm[j] = spins_mm[j] - delta;
					spins_pm[i] = spins_mm[i] + delta;
					spins_pm[j] = spins_mm[j] - delta;
					spins_mp[i] = spins_mm[i] - delta;
					spins_mp[j] = spins_mm[j] + delta;

					Utility::Vectormath::Normalize_3Nos(spins_pp);
					Utility::Vectormath::Normalize_3Nos(spins_mm);
					Utility::Vectormath::Normalize_3Nos(spins_pm);
					Utility::Vectormath::Normalize_3Nos(spins_mp);

					double E_pp = this->Energy(spins_pp);
					double E_mm = this->Energy(spins_mm);
					double E_pm = this->Energy(spins_pm);
					double E_mp = this->Energy(spins_mp);

					hessian[i * 3 * nos + j] = (E_pp - E_pm - E_mp + E_mm) / (4 * delta*delta);
				}
			}
		}
    }

    void Hamiltonian::Effective_Field(const std::vector<double> & spins, std::vector<double> & field)
    {
		// This is a regular finite difference implementation (probably not very efficient)

        int nos = spins.size()/3;

		// Calculate finite difference
		std::vector<double> spins_plus(3 * nos, 0);
		std::vector<double> spins_minus(3 * nos, 0);

		for (int i = 0; i < 3 * nos; ++i)
		{
			spins_plus = spins;
			spins_minus = spins;
			spins_plus[i] = spins_plus[i] + delta;
			spins_minus[i] = spins_minus[i] - delta;

			Utility::Vectormath::Normalize_3Nos(spins_plus);
			Utility::Vectormath::Normalize_3Nos(spins_minus);

			double E_plus = this->Energy(spins_plus);
			double E_minus = this->Energy(spins_minus);

			field[i] = (E_minus - E_plus) / (2 * delta);
		}
    }

    double Hamiltonian::Energy(const std::vector<double> & spins)
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Energy() of the Hamiltonian base class!"));
        throw Utility::Exception::Not_Implemented;
        return 0.0;
    }

    std::vector<std::vector<double>> Hamiltonian::Energy_Array_per_Spin(const std::vector<double> & spins)
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Energy_Array_per_Spin() of the Hamiltonian base class!"));
        throw Utility::Exception::Not_Implemented;
        return std::vector<std::vector<double>>(spins.size(), std::vector<double>(7, 0.0));
    }

    std::vector<double> Hamiltonian::Energy_Array(const std::vector<double> & spins)
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Energy_Array() of the Hamiltonian base class!"));
        throw Utility::Exception::Not_Implemented;
        return std::vector<double>(7, 0.0);
    }

    std::string Hamiltonian::Name()
    {
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Name() of the Hamiltonian base class!"));
        return "--";
    }
}