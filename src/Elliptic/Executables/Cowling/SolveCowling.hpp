// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/RadiallyCompressedCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/Actions/InitializeFields.hpp"
#include "Elliptic/Actions/RunEventsAndTriggers.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/ApplyOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/InitializeDomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/InitializeSubdomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/Systems/Cowling/Actions/CheckConvergence.hpp"
#include "Elliptic/Systems/Cowling/Actions/InitializeFixedSources.hpp"
#include "Elliptic/Systems/Cowling/Actions/IterativeSolve.hpp"
#include "Elliptic/Systems/Cowling/Actions/ProcessVolumeData.hpp"
#include "Elliptic/Systems/Cowling/BoundaryConditions/Factory.hpp"
#include "Elliptic/Systems/Cowling/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Cowling/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "Elliptic/Triggers/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/Actions/ReceiveVolumeData.hpp"
#include "IO/Importers/Actions/RegisterWithElementDataReader.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/RandomizeVariables.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/BuildMatrix.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/MakeIdentityIfSkipped.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Multigrid.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Actions/CommunicateOverlapFields.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/Binary.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Cowling/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Factory.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

/// \cond

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace SolveCowling::OptionTags {
struct LinearSolverGroup {
  static std::string name() { return "LinearSolver"; }
  static constexpr Options::String help =
      "The iterative Krylov-subspace linear solver";
};
struct GmresGroup {
  static std::string name() { return "GMRES"; }
  static constexpr Options::String help = "Options for the GMRES linear solver";
  using group = LinearSolverGroup;
};
struct SchwarzSmootherGroup {
  static std::string name() { return "SchwarzSmoother"; }
  static constexpr Options::String help = "Options for the Schwarz smoother";
  using group = LinearSolverGroup;
};
struct MultigridGroup {
  static std::string name() { return "Multigrid"; }
  static constexpr Options::String help = "Options for the multigrid";
  using group = LinearSolverGroup;
};

struct SelfConsistentGroup {
  static std::string name() { return "SelfConsistent"; }
  static constexpr Options::String help =
      "Options for the self-consistent iteration";
};

}  // namespace SolveCowling::OptionTags

struct OptionsGroup {
  static std::string name() { return "Importers"; }
  static constexpr Options::String help = "Numeric volume data";
};

/// \cond
struct Metavariables {
  static constexpr size_t volume_dim = 3;
  using system = Cowling::FirstOrderSystem;

  using background_tag =
      elliptic::Tags::Background<elliptic::analytic_data::Background>;
  using initial_guess_tag =
      elliptic::Tags::InitialGuess<elliptic::analytic_data::InitialGuess>;

  static constexpr Options::String help{
      "Find the solution to the Cowling problem."};

  // These are the fields we solve for
  using fields_tag = ::Tags::Variables<typename system::primal_fields>;
  // These are the fixed sources, i.e. the RHS of the equations
  using fixed_sources_tag = db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
  // This is the linear operator applied to the fields. We'll only use it to
  // apply the operator to the initial guess, so an optimization would be to
  // re-use the `operator_applied_to_vars_tag` below. This optimization needs a
  // few minor changes to the parallel linear solver algorithm.
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;

  // The linear solver algorithm. We must use GMRES since the operator is
  // not guaranteed to be symmetric. It can be made symmetric by multiplying by
  // the DG mass matrix.
  using linear_solver = LinearSolver::gmres::Gmres<
      Metavariables, fields_tag, SolveCowling::OptionTags::LinearSolverGroup,
      true, fixed_sources_tag, LinearSolver::multigrid::Tags::IsFinestGrid>;
  using linear_solver_iteration_id =
      Convergence::Tags::IterationId<typename linear_solver::options_group>;

  using self_consistent_iteration_id = Cowling::Tags::SolveIteration;

  // Precondition each linear solver iteration with a multigrid V-cycle
  using multigrid = LinearSolver::multigrid::Multigrid<
      volume_dim, typename linear_solver::operand_tag,
      SolveCowling::OptionTags::MultigridGroup, elliptic::dg::Tags::Massive,
      typename linear_solver::preconditioner_source_tag>;
  // Smooth each multigrid level with a number of Schwarz smoothing steps
  using subdomain_operator =
      elliptic::dg::subdomain_operator::SubdomainOperator<
          system, SolveCowling::OptionTags::SchwarzSmootherGroup>;
  using schwarz_smoother = LinearSolver::Schwarz::Schwarz<
      typename multigrid::smooth_fields_tag,
      SolveCowling::OptionTags::SchwarzSmootherGroup, subdomain_operator,
      tmpl::list<>, typename multigrid::smooth_source_tag,
      LinearSolver::multigrid::Tags::MultigridLevel>;
  // For the GMRES linear solver we need to apply the DG operator to its
  // internal "operand" in every iteration of the algorithm.
  using vars_tag = typename linear_solver::operand_tag;
  using operator_applied_to_vars_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, vars_tag>;
  // We'll buffer the corresponding fluxes in this tag, but won't actually need
  // to access them outside applying the operator
  using fluxes_vars_tag =
      ::Tags::Variables<db::wrap_tags_in<LinearSolver::Tags::Operand,
                                         typename system::primal_fluxes>>;

  using analytic_solution_fields = typename system::primal_fields;
  using observe_fields = tmpl::append<
      analytic_solution_fields,
      tmpl::list<domain::Tags::Coordinates<volume_dim, Frame::Inertial>,
                 domain::Tags::RadiallyCompressedCoordinatesCompute<
                     volume_dim, Frame::Inertial>,
                 ::Tags::FixedSource<::CurvedScalarWave::Tags::Psi>,
                 gr::Tags::WeylElectricScalar<DataVector>,
                 gr::Tags::WeylMagneticScalar<DataVector>>>;
  using observer_compute_tags =
      tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                 ::Events::Tags::ObserverDetInvJacobianCompute<
                     Frame::ElementLogical, Frame::Inertial>,
                 ::Tags::DerivTensorCompute<
                     ::CurvedScalarWave::Tags::Psi,
                     domain::Tags::InverseJacobian<
                         volume_dim, Frame::ElementLogical, Frame::Inertial>,
                     ::domain::Tags::Mesh<volume_dim>>,
                 Cowling::Tags::MoveDerivToPhi, Cowling::Tags::UpdatePi>;

  // Collect all items to store in the cache.
  using const_global_cache_tags =
      tmpl::list<background_tag, initial_guess_tag,
                 domain::Tags::RadiallyCompressedCoordinatesOptions,
                 Cowling::Tags::MaxIterations, Cowling::Tags::Epsilon1,
                 Cowling::Tags::Epsilon2, Cowling::Tags::Epsilon4,
                 Cowling::Tags::DampingParameter>;

  using analytic_solutions_and_data = tmpl::push_back<
      Cowling::Solutions::all_analytic_solutions,
      Xcts::AnalyticData::Binary<elliptic::analytic_data::AnalyticSolution,
                                 Cowling::Solutions::all_analytic_solutions>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<elliptic::analytic_data::Background,
                   analytic_solutions_and_data>,
        tmpl::pair<elliptic::analytic_data::InitialGuess,
                   Cowling::Solutions::all_initial_guesses<volume_dim>>,
        tmpl::pair<elliptic::analytic_data::AnalyticSolution,
                   Cowling::Solutions::all_analytic_solutions>,
        tmpl::pair<elliptic::BoundaryConditions::BoundaryCondition<volume_dim>,
                   Cowling::BoundaryConditions::standard_boundary_conditions>,
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<
                       Events::Completion,
                       dg::Events::field_observations<
                           volume_dim, observe_fields, observer_compute_tags,
                           LinearSolver::multigrid::Tags::IsFinestGrid>>>>,
        tmpl::pair<Trigger, elliptic::Triggers::all_triggers<
                                typename linear_solver::options_group>>,
        tmpl::pair<PhaseChange, tmpl::list<PhaseControl::VisitAndReturn<
                                    Parallel::Phase::BuildMatrix>>>>;
  };

  // Collect all reduction tags for observers
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<typename factory_creation::factory_classes, Event>,
          linear_solver, multigrid, schwarz_smoother>>>;

  // For labeling the yaml option for RandomizeVariables
  struct RandomizeInitialGuess {};

  using initialization_actions = tmpl::list<
      elliptic::dg::Actions::InitializeDomain<volume_dim>,
      typename linear_solver::initialize_element,
      typename multigrid::initialize_element,
      typename schwarz_smoother::initialize_element,
      elliptic::Actions::InitializeFields<system, initial_guess_tag>,
      Actions::RandomizeVariables<
          ::Tags::Variables<typename system::primal_fields>,
          RandomizeInitialGuess>,
      Cowling::Actions::InitializeFixedSources<system, initial_guess_tag>,
      Parallel::Actions::TerminatePhase>;

  using build_linear_operator_actions = elliptic::dg::Actions::apply_operator<
      system, true, linear_solver_iteration_id, vars_tag, fluxes_vars_tag,
      operator_applied_to_vars_tag>;

  using register_actions =
      tmpl::list<observers::Actions::RegisterEventsWithObservers,
                 typename schwarz_smoother::register_element,
                 typename multigrid::register_element,
                 importers::Actions::RegisterWithElementDataReader,
                 LinearSolver::Actions::build_matrix_register<
                     LinearSolver::multigrid::Tags::IsFinestGrid>,
                 Parallel::Actions::TerminatePhase>;

  template <typename Label>
  using smooth_actions =
      typename schwarz_smoother::template solve<build_linear_operator_actions,
                                                Label>;

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
      domain::make_faces_tags<
          3, tmpl::list<gr::Tags::Lapse<DataVector>,
                        gr::Tags::Shift<DataVector, 3>,
                        Xcts::Tags::ConformalFactor<DataVector>>>>>;

  using import_actions = tmpl::list<
      importers::Actions::ReadVolumeData<OptionsGroup, import_fields>,
      Cowling::Actions::ProcessVolumeData<import_fields>,
      elliptic::dg::Actions::initialize_operator<system, background_tag>,
      elliptic::dg::subdomain_operator::Actions::InitializeSubdomain<
          system, background_tag, typename schwarz_smoother::options_group>,
      LinearSolver::Schwarz::Actions::SendOverlapFields<
          communicated_overlap_tags, typename schwarz_smoother::options_group,
          false>,
      LinearSolver::Schwarz::Actions::ReceiveOverlapFields<
          volume_dim, communicated_overlap_tags,
          typename schwarz_smoother::options_group>,
      Parallel::Actions::TerminatePhase>;

  using solve_actions = tmpl::list<
      PhaseControl::Actions::ExecutePhaseChange,
      Cowling::Actions::IterativeSolve,
      LinearSolver::Schwarz::Actions::SendOverlapFields<
          tmpl::list<fixed_sources_tag>,
          typename schwarz_smoother::options_group, false>,
      LinearSolver::Schwarz::Actions::ReceiveOverlapFields<
          volume_dim, tmpl::list<fixed_sources_tag>,
          typename schwarz_smoother::options_group>,
      elliptic::dg::Actions::ImposeInhomogeneousBoundaryConditionsOnSource<
          system, fixed_sources_tag>,
      elliptic::dg::Actions::apply_operator<
          system, true, linear_solver_iteration_id, fields_tag, fluxes_vars_tag,
          operator_applied_to_fields_tag, vars_tag, fluxes_vars_tag>,
      typename linear_solver::template solve<tmpl::list<
          typename multigrid::template solve<
              build_linear_operator_actions,
              smooth_actions<LinearSolver::multigrid::VcycleDownLabel>,
              smooth_actions<LinearSolver::multigrid::VcycleUpLabel>>,
          ::LinearSolver::Actions::make_identity_if_skipped<
              multigrid, build_linear_operator_actions>>>,
      elliptic::Actions::RunEventsAndTriggers<self_consistent_iteration_id>,
      Cowling::Actions::CheckConvergence<typename linear_solver::options_group>,
      Parallel::Actions::TerminatePhase>;

  using build_matrix_actions = tmpl::list<
      LinearSolver::Actions::build_matrix_actions<
          linear_solver_iteration_id, vars_tag, operator_applied_to_vars_tag,
          build_linear_operator_actions,
          domain::Tags::Coordinates<volume_dim, Frame::Inertial>,
          LinearSolver::multigrid::Tags::IsFinestGrid>,
      Parallel::Actions::TerminatePhase>;

  using dg_element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,
          Parallel::PhaseActions<Parallel::Phase::Register, register_actions>,
          Parallel::PhaseActions<Parallel::Phase::ImportInitialData,
                                 import_actions>,
          Parallel::PhaseActions<Parallel::Phase::Solve, solve_actions>,
          Parallel::PhaseActions<Parallel::Phase::BuildMatrix,
                                 build_matrix_actions>>,
      LinearSolver::multigrid::ElementsAllocator<
          volume_dim, typename multigrid::options_group>>;
  // Specify all parallel components that will execute actions at some point.
  using component_list = tmpl::flatten<
      tmpl::list<dg_element_array, typename linear_solver::component_list,
                 typename multigrid::component_list,
                 typename schwarz_smoother::component_list,
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
