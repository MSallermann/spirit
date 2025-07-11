# Methods
This section provides a high-level overview of the implemented methods and equations.

For basic usage examples, see the pages for the
 - [C API](core/docs/C_API.md)
 - [Python API](core/docs/Python_API.md)

## The Landau-Lifshitz-Gilbert (LLG) equation
The LLG equation can be used to simulate the **dynamics** of a spin system 

- at finite temperatures (using the Langevin thermostat)
- with spin polarized currents [spin transfer torque (SOT) and spin orbit torque (SOT)]

The full form implemented in Spirit is

![Equation](https://math.vercel.app/?bgcolor=auto&from=%5Cbegin%7Baligned%7D%0A%09%5Cfrac%7B%5Cpartial%20%5Cvec%7Bn%7D_i%7D%7B%5Cpartial%20t%7D%20%3D%20-%26%5Cfrac%7B%5Cgamma%7D%7B%281%2B%5Calpha%5E2%29%5Cmu_i%7D%20%5Cvec%7Bn%7D_i%20%5Ctimes%20%20%5Cvec%7BB%7D%5E%5Cmathrm%7Beff%7D_i%5C%5C%0A%09-%20%26%5Cfrac%7B%5Cgamma%20%5Calpha%7D%7B%281%2B%5Calpha%5E2%29%5Cmu_i%7D%5Cvec%7Bn%7D_i%20%5Ctimes%20%28%5Cvec%7Bn%7D_i%20%5Ctimes%20%5Cvec%7BB%7D%5E%5Cmathrm%7Beff%7D_i%29%5C%5C%0A%09-%20%26%5Cdfrac%7B%5Calpha-%5Cbeta%7D%7B%281%2B%5Calpha%5E2%29%7D%20u%20%5Cvec%7Bn%7D_i%20%5Ctimes%20%28%5Cvec%7Bj%7D_e%20%5Ccdot%20%5Cnabla_%7B%5Cvec%7Br%7D%7D%20%29%5Cvec%7Bn%7D_i%5C%5C%0A%09%2B%20%26%5Cdfrac%7B1%2B%5Cbeta%20%5Calpha%7D%7B%281%2B%5Calpha%5E2%29%7D%20u%20%5Cvec%7Bn%7D_i%20%5Ctimes%20%28%5Cvec%7Bn%7D_i%20%5Ctimes%20%28%5Cvec%7Bj%7D_e%20%5Ccdot%20%5Cnabla_%7B%5Cvec%7Br%7D%7D%20%29%5Cvec%7Bn%7D_i%0A%5Cend%7Baligned%7D.svg)


### Energy minimization

Energy minimization is invoked by using the LLG method with any of the "minimization" solvers (see table below)

| Dynamics solvers                | Minimization Solvers                       |
| ------------------------------- | ------------------------------------------ |
| `Depondt`, `Heun`, `SIB`, `RK4` | `VP`, `VP_OSO`, `LBFGS_Atlas`, `LBFGS_OSO` |

:::{caution}
Using the LLG method with any of the "minimization" solvers does, in fact, **not** solve the LLG equation. 
The only purpose of these solvers is to find a stationary configuration, where the energy gradients (which form the dissipative contribution to the LLG) vanish.
:::

:::{info}
A good default approach is to start with a minization with a few iterations of the `VP` solver and then switch to one of the more powerful `LBFGS` solvers.
:::

:::{warning}
While possible, the following is **not recommended** in practice:

The `direct_minimization` attribute can be used to deactivate all non-dissipative contributions to the LLG, then an energy minimization can be performed by following the resulting trajectory with any of the dynamics solvers.

```C++
#include <Spirit/Parameters.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <memory>

auto state = std::shared_ptr<State>(State_Setup("input/input.cfg"), State_Delete);
Parameters_LLG_Set_Direct_Minimization(state.get(), true);
Simulation_LLG_Start(state.get(), Solver_Depondt);
```
:::


## The geodesic nudged elastic band (GNEB) method

## The Monte Carlo (MC) method

## The eigenmode analysis (EMA) method

## Minimum mode following (MMF) method

## Harmonic Transition state theory (HTST)
