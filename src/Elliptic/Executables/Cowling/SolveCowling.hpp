// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "Domain/Creators/Factory3D.hpp"
#include "Domain/RadiallyCompressedCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/RunEventsAndTriggers.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/Executables/Solver.hpp"
#include "Elliptic/Systems/Cowling/Actions/ProcessVolumeData.hpp"
#include "Elliptic/Systems/Cowling/BoundaryConditions/Factory.hpp"
#include "Elliptic/Systems/Cowling/FirstOrderSystem.hpp"
#include "Elliptic/Triggers/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/Actions/ReceiveVolumeData.hpp"
#include "IO/Importers/Actions/RegisterWithElementDataReader.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Amr/Actions/SendAmrDiagnostics.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Factory.hpp"
#include "ParallelAlgorithms/Amr/Projectors/DefaultInitialize.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "ParallelAlgorithms/Amr/Tags.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Actions/CommunicateOverlapFields.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Cowling/Factory.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

struct OptionsGroup {
  static std::string name() { return "Importers"; }
  static constexpr Options::String help = "Numeric volume data";
};

/// \cond
struct Metavariables {
  static constexpr Options::String help{"Solve for sGB initial data"};

  static constexpr size_t volume_dim = 3;
  using system = Cowling::FirstOrderSystem;
  using solver = elliptic::Solver<Metavariables>;

  using observe_fields = tmpl::append<
      typename system::primal_fields, typename system::background_fields,
      typename solver::observe_fields,
      tmpl::list<domain::Tags::Coordinates<volume_dim, Frame::Inertial>,
                 domain::Tags::RadiallyCompressedCoordinatesCompute<
                     volume_dim, Frame::Inertial>>>;
  using observer_compute_tags =
      tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                 ::Events::Tags::ObserverDetInvJacobianCompute<
                     Frame::ElementLogical, Frame::Inertial>>;
  using analytic_solutions_and_data = tmpl::push_back<
      Cowling::Solutions::all_analytic_solutions,
      Xcts::AnalyticData::Binary<elliptic::analytic_data::Background,
                                 Cowling::Solutions::all_analytic_solutions>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<elliptic::analytic_data::Background,
                   analytic_solutions_and_data>,
        tmpl::pair<elliptic::analytic_data::InitialGuess,
                   Cowling::Solutions::all_initial_guesses<volume_dim>>,
        tmpl::pair<elliptic::BoundaryConditions::BoundaryCondition<volume_dim>,
                   Cowling::BoundaryConditions::standard_boundary_conditions>,
        tmpl::pair<elliptic::analytic_data::AnalyticSolution, tmpl::list<>>,
        tmpl::pair<::amr::Criterion,
                   ::amr::Criteria::standard_criteria<
                       volume_dim, tmpl::list<::CurvedScalarWave::Tags::Psi>>>,
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<
                       Events::Completion,
                       dg::Events::field_observations<
                           volume_dim, observe_fields, observer_compute_tags,
                           LinearSolver::multigrid::Tags::IsFinestGrid>>>>,
        tmpl::pair<Trigger, elliptic::Triggers::all_triggers<
                                ::amr::OptionTags::AmrGroup>>,
        tmpl::pair<
            PhaseChange,
            tmpl::list<
                // Phase for building a matrix representation of the
                // operator
                PhaseControl::VisitAndReturn<Parallel::Phase::BuildMatrix>,
                // Phases for AMR
                PhaseControl::VisitAndReturn<
                    Parallel::Phase::EvaluateAmrCriteria>,
                PhaseControl::VisitAndReturn<Parallel::Phase::AdjustDomain>,
                PhaseControl::VisitAndReturn<Parallel::Phase::CheckDomain>>>>;
  };

  // Additional items to store in the global cache
  using const_global_cache_tags =
      tmpl::list<domain::Tags::RadiallyCompressedCoordinatesOptions,
                 Cowling::Tags::Epsilon2, Cowling::Tags::Epsilon4>;

  // Collect all reduction tags for observers
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::push_back<
          tmpl::at<factory_creation::factory_classes, Event>, solver>>;

  using initialization_actions =
      tmpl::push_back<typename solver::initialization_actions,
                      Parallel::Actions::TerminatePhase>;

  using register_actions =
      tmpl::push_back<typename solver::register_actions,
                      observers::Actions::RegisterEventsWithObservers>;

  using import_fields =
      tmpl::list<gr::Tags::SpatialMetric<DataVector, volume_dim>,
                 gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<DataVector, volume_dim>,
                 gr::Tags::ExtrinsicCurvature<DataVector, volume_dim>,
                 Xcts::Tags::ConformalFactor<DataVector>,
                 Xcts::Tags::InverseConformalMetric<DataVector, volume_dim,
                                                    Frame::Inertial>>;
  using communicated_overlap_tags = tmpl::flatten<tmpl::list<
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, volume_dim>,
      Xcts::Tags::ConformalFactor<DataVector>,
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>,
                    tmpl::size_t<volume_dim>, Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<volume_dim>,
                    Frame::Inertial>,
      Cowling::Tags::Epsilon2, Cowling::Tags::Epsilon4,
      gr::Tags::WeylElectricScalar<DataVector>,
      gr::Tags::WeylMagneticScalar<DataVector>,
      domain::make_faces_tags<
          3, tmpl::list<gr::Tags::Lapse<DataVector>,
                        gr::Tags::Shift<DataVector, 3>,
                        Xcts::Tags::ConformalFactor<DataVector>>>>>;

  using import_actions = tmpl::list<
      importers::Actions::ReadVolumeData<OptionsGroup, import_fields>,
      Cowling::Actions::ProcessVolumeData<import_fields>,
      LinearSolver::Schwarz::Actions::SendOverlapFields<
          communicated_overlap_tags,
          typename solver::schwarz_smoother::options_group, false>,
      LinearSolver::Schwarz::Actions::ReceiveOverlapFields<
          volume_dim, communicated_overlap_tags,
          typename solver::schwarz_smoother::options_group, false>,
      Parallel::Actions::TerminatePhase>;

  using solve_actions = typename solver::template solve_actions<tmpl::list<>>;

  using dg_element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,
          Parallel::PhaseActions<
              Parallel::Phase::Register,
              tmpl::push_back<register_actions,
                              importers::Actions::RegisterWithElementDataReader,
                              Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<Parallel::Phase::ImportInitialData,
                                 import_actions>,
          Parallel::PhaseActions<Parallel::Phase::Solve, solve_actions>>,
      LinearSolver::multigrid::ElementsAllocator<
          volume_dim, typename solver::multigrid::options_group>>;

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using element_array = dg_element_array;
    using projectors = tmpl::push_back<
        typename solver::amr_projectors,
        ::amr::projectors::DefaultInitialize<
            gr::Tags::WeylElectricScalar<DataVector>,
            gr::Tags::WeylMagneticScalar<DataVector>,
            gr::Tags::Lapse<DataVector>,
            Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>,
            gr::Tags::Shift<DataVector, 3, Frame::Inertial>,
            Xcts::Tags::ConformalFactor<DataVector>,
            Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>,
            domain::Tags::Faces<3, gr::Tags::Lapse<DataVector>>,
            domain::Tags::Faces<
                3, gr::Tags::Shift<DataVector, 3, Frame::Inertial>>,
            domain::Tags::Faces<3, Xcts::Tags::ConformalFactor<DataVector>>,
            LinearSolver::Schwarz::Tags::Overlaps<
                gr::Tags::Lapse<DataVector>, 3,
                elliptic::OptionTags::SchwarzSmootherGroup>,
            LinearSolver::Schwarz::Tags::Overlaps<
                gr::Tags::Shift<DataVector, 3, Frame::Inertial>, 3,
                elliptic::OptionTags::SchwarzSmootherGroup>,
            LinearSolver::Schwarz::Tags::Overlaps<
                Xcts::Tags::ConformalFactor<DataVector>, 3,
                elliptic::OptionTags::SchwarzSmootherGroup>,
            LinearSolver::Schwarz::Tags::Overlaps<
                Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>,
                            tmpl::size_t<3>, Frame::Inertial>,
                3, elliptic::OptionTags::SchwarzSmootherGroup>,
            LinearSolver::Schwarz::Tags::Overlaps<
                Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                            Frame::Inertial>,
                3, elliptic::OptionTags::SchwarzSmootherGroup>,
            LinearSolver::Schwarz::Tags::Overlaps<
                Cowling::Tags::Epsilon2, 3,
                elliptic::OptionTags::SchwarzSmootherGroup>,
            LinearSolver::Schwarz::Tags::Overlaps<
                Cowling::Tags::Epsilon4, 3,
                elliptic::OptionTags::SchwarzSmootherGroup>,
            LinearSolver::Schwarz::Tags::Overlaps<
                gr::Tags::WeylElectricScalar<DataVector>, 3,
                elliptic::OptionTags::SchwarzSmootherGroup>,
            LinearSolver::Schwarz::Tags::Overlaps<
                gr::Tags::WeylMagneticScalar<DataVector>, 3,
                elliptic::OptionTags::SchwarzSmootherGroup>,
            LinearSolver::Schwarz::Tags::Overlaps<
                domain::Tags::Faces<3, gr::Tags::Lapse<DataVector>>, 3,
                elliptic::OptionTags::SchwarzSmootherGroup>,
            LinearSolver::Schwarz::Tags::Overlaps<
                domain::Tags::Faces<
                    3, gr::Tags::Shift<DataVector, 3, Frame::Inertial>>,
                3, elliptic::OptionTags::SchwarzSmootherGroup>,
            LinearSolver::Schwarz::Tags::Overlaps<
                domain::Tags::Faces<3, Xcts::Tags::ConformalFactor<DataVector>>,
                3, elliptic::OptionTags::SchwarzSmootherGroup>>>;
  };

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<dg_element_array, register_actions>>;
  };

  // Specify all parallel components that will execute actions at some point.
  using component_list = tmpl::flatten<
      tmpl::list<dg_element_array, typename solver::component_list,
                 observers::Observer<Metavariables>,
                 observers::ObserverWriter<Metavariables>,
                 importers::ElementDataReader<Metavariables>>>;

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Register,
       Parallel::Phase::ImportInitialData, Parallel::Phase::Solve,
       Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
/// \endcond
