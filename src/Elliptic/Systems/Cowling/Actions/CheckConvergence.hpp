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
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Cowling/Actions/IterativeSolve.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Cowling::Actions {

/*!
 * \brief Checks convergence of self-consistent solves.
 *
 * This actions decides whether to repeat the self-consistent solve again, or
 * stop. Stopping conditions are that the previous solution solves the updated
 * elliptic problem, or that the self-consistent solve iteration has passed some
 * user-defined value. Otherwise, repeat the algorithm. Must be used in
 * conjuction with IterativeSolve.
 *
 * Uses:
 * - DataBox:
 *   - `Poisson::Tags::SolveIteration`
 *   - `Poisson::Tags::MaxIterations`
 *   - `Convergence::Tags::IterationId<OptionsGroup>`
 */

template <typename OptionsGroup>
struct CheckConvergence {
 private:
 public:
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (db::get<Convergence::Tags::IterationId<OptionsGroup>>(box) == 0 ||
        db::get<Cowling::Tags::SolveIteration>(box) >=
            db::get<Cowling::Tags::MaxIterations>(box)) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    } else {
      return {Parallel::AlgorithmExecution::Continue,
              tmpl::index_of<ActionList, IterativeSolve>::value};
    }
  }
};

}  // namespace Cowling::Actions
