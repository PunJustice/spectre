// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Actions {
/*!
 * \brief Invokes all events specified in `Tags::EventsRunAtCleanup`.
 *
 * Before running the events, floating point exceptions are disabled. This is
 * to allow manipulating data even if there are `NaN` or other problematic
 * values. We ultimately just want to be able to see the state of the
 * simulation at failure.
 *
 * This action is intended to be executed in the
 * `Parallel::Phase::PostFailureCleanup` phase.
 *
 * \note The simulation will almost certainly fail with different
 * elements at different times.
 */
template <typename ObservationId>
struct RunEventsOnFailure {
 private:
  template <typename Event>
  struct get_tags {
    using type = typename Event::compute_tags_for_observation_box;
  };

 public:
  using const_global_cache_tags =
      tmpl::list<::Tags::EventsRunAtCleanup,
                 ::Tags::EventsRunAtCleanupObservationValue>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const component) {
    // We explicitly disable FPEs because we are dumping during a failure and
    // so can't rely on the data being safe.
    disable_floating_point_exceptions();

    const Event::ObservationValue observation_value{
        db::tag_name<ObservationId>(),
        db::get<Tags::EventsRunAtCleanupObservationValue>(box)};

    using compute_tags = tmpl::remove_duplicates<tmpl::filter<
        tmpl::flatten<tmpl::transform<
            tmpl::at<typename Metavariables::factory_creation::factory_classes,
                     Event>,
            get_tags<tmpl::_1>>>,
        db::is_compute_tag<tmpl::_1>>>;
    std::optional observation_box{
        make_observation_box<compute_tags>(make_not_null(&box))};

    for (const auto& event : db::get<::Tags::EventsRunAtCleanup>(box)) {
      event->run(observation_box.value(), cache, array_index, component,
                 observation_value);
    }

    // Do not re-enable FPEs because other parts of the pipeline might rely on
    // them being disabled. We generally have them disabled during cleanup.
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
