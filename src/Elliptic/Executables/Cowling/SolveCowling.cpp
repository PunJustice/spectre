// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Executables/Cowling/SolveCowling.hpp"

#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Parallel/CharmMain.tpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<Metavariables>();
  Parallel::charmxx::register_init_node_and_proc(
      {&domain::creators::register_derived_with_charm,
       &register_derived_classes_with_charm<
           Metavariables::schwarz_smoother::subdomain_solver>,
       &register_factory_classes_with_charm<Metavariables>},
      {});
}
