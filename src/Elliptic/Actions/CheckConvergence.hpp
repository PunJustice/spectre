// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "Elliptic/Utilities/GetAnalyticData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ApplyMassMatrix.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace elliptic::Actions {

/*!
 * \brief Keeps track of how many iterative solves have occured.
 *
 * This action simply increments a double k, tracking how many iterations have
 * occured. If this exceeds some predefined value, stop the solve. Otherwise,
 * repeat the algorithm.
 *
 * Uses:
 * - DataBox:
 *   - `Poisson::Tags::SolveIteration`
 */
template <typename System, typename BackgroundTag>
struct CheckConvergence {
 private:
 public:
  using simple_tags = Poisson::Tags::SolveIteration;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (db::get<Poisson::Tags::SolveIteration>(box) == 5) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    } else {
          return {Parallel::AlgorithmExecution::Pause,
            tmpl::index_of<ActionList, IterativeSolve>::value + 1};
    }
  }
};

}  // namespace elliptic::Actions
