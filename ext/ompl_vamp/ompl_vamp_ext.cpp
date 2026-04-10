/**
 * OMPL + VAMP C++ extension for Autolife planning.
 *
 * Instantiates OMPL planners with VAMP's SIMD-accelerated collision
 * checking for the Autolife robot.  The entire planning pipeline runs
 * in C++ — Python only crosses the boundary once per plan() call.
 *
 * Supports both full-body (24 DOF) and subgroup planning (reduced
 * state space with frozen joints expanded before collision checks).
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// OMPL — core
#include <ompl/base/Constraint.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/constraint/AtlasStateSpace.h>
#include <ompl/base/spaces/constraint/ConstrainedStateSpace.h>
#include <ompl/base/spaces/constraint/ProjectedStateSpace.h>
#include <ompl/base/spaces/constraint/TangentBundleStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
// OMPL — informed trees (asymptotically optimal)
#include <ompl/geometric/planners/informedtrees/ABITstar.h>
#include <ompl/geometric/planners/informedtrees/AITstar.h>
#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/geometric/planners/informedtrees/EITstar.h>
#include <ompl/geometric/planners/lazyinformedtrees/BLITstar.h>
// OMPL — FMT
#include <ompl/geometric/planners/fmt/BFMT.h>
#include <ompl/geometric/planners/fmt/FMT.h>
// OMPL — KPIECE
#include <ompl/geometric/planners/kpiece/BKPIECE1.h>
#include <ompl/geometric/planners/kpiece/KPIECE1.h>
#include <ompl/geometric/planners/kpiece/LBKPIECE1.h>
// OMPL — PRM
#include <ompl/geometric/planners/prm/LazyPRM.h>
#include <ompl/geometric/planners/prm/LazyPRMstar.h>
#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/geometric/planners/prm/SPARS.h>
#include <ompl/geometric/planners/prm/SPARStwo.h>
// OMPL — RRT variants
#include <ompl/geometric/planners/rrt/BiTRRT.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/geometric/planners/rrt/LBTRRT.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/RRTXstatic.h>
#include <ompl/geometric/planners/rrt/RRTsharp.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/STRRTstar.h>
#include <ompl/geometric/planners/rrt/TRRT.h>
// OMPL — exploration-based
#include <ompl/geometric/planners/est/BiEST.h>
#include <ompl/geometric/planners/est/EST.h>
#include <ompl/geometric/planners/pdst/PDST.h>
#include <ompl/geometric/planners/sbl/SBL.h>
#include <ompl/geometric/planners/stride/STRIDE.h>

// VAMP (header-only — our fork with Autolife robot)
#include <vamp/collision/environment.hh>
#include <vamp/collision/shapes.hh>
#include <vamp/planning/validate.hh>
#include <vamp/robots/autolife.hh>

namespace nb = nanobind;
namespace ob = ompl::base;
namespace og = ompl::geometric;

using Robot = vamp::robots::Autolife;
static constexpr std::size_t kRake = vamp::FloatVectorWidth;
using VampEnv = vamp::collision::Environment<vamp::FloatVector<kRake>>;
using FloatEnv = vamp::collision::Environment<float>;

// ─── OMPL ↔ VAMP validity checker (full-body) ───────────────────────

class AutolifeValidityChecker : public ob::StateValidityChecker {
 public:
  AutolifeValidityChecker(const ob::SpaceInformationPtr &si, const VampEnv &env)
      : ob::StateValidityChecker(si), env_(env) {}

  auto isValid(const ob::State *state) const -> bool override {
    auto config = ompl_to_vamp(state);
    return vamp::planning::validate_motion<Robot, kRake, 1>(config, config,
                                                            env_);
  }

 private:
  const VampEnv &env_;

  static auto ompl_to_vamp(const ob::State *state) -> Robot::Configuration {
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars>
            buf{};
    const auto *rv = state->as<ob::RealVectorStateSpace::StateType>();
    for (std::size_t i = 0; i < Robot::dimension; ++i)
      buf[i] = static_cast<float>(rv->values[i]);
    return Robot::Configuration(buf.data());
  }
};

class AutolifeMotionValidator : public ob::MotionValidator {
 public:
  AutolifeMotionValidator(const ob::SpaceInformationPtr &si, const VampEnv &env)
      : ob::MotionValidator(si), env_(env) {}

  auto checkMotion(const ob::State *s1, const ob::State *s2) const
      -> bool override {
    return vamp::planning::validate_motion<Robot, kRake, Robot::resolution>(
        ompl_to_vamp(s1), ompl_to_vamp(s2), env_);
  }

  auto checkMotion(const ob::State *s1, const ob::State *s2,
                   std::pair<ob::State *, double> &last_valid) const
      -> bool override {
    last_valid.first = nullptr;
    last_valid.second = 0.0;
    return checkMotion(s1, s2);
  }

 private:
  const VampEnv &env_;

  static auto ompl_to_vamp(const ob::State *state) -> Robot::Configuration {
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars>
            buf{};
    const auto *rv = state->as<ob::RealVectorStateSpace::StateType>();
    for (std::size_t i = 0; i < Robot::dimension; ++i)
      buf[i] = static_cast<float>(rv->values[i]);
    return Robot::Configuration(buf.data());
  }
};

// ─── Subgroup validity checker (reduced dim → full config → VAMP) ───

class SubgroupValidityChecker : public ob::StateValidityChecker {
 public:
  SubgroupValidityChecker(const ob::SpaceInformationPtr &si, const VampEnv &env,
                          std::vector<int> active_indices,
                          std::vector<float> frozen_config)
      : ob::StateValidityChecker(si),
        env_(env),
        active_(std::move(active_indices)),
        frozen_(std::move(frozen_config)) {}

  auto isValid(const ob::State *state) const -> bool override {
    auto config = expand(state);
    return vamp::planning::validate_motion<Robot, kRake, 1>(config, config,
                                                            env_);
  }

 private:
  const VampEnv &env_;
  std::vector<int> active_;
  std::vector<float> frozen_;

  auto expand(const ob::State *state) const -> Robot::Configuration {
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars>
            buf{};
    std::copy(frozen_.begin(), frozen_.end(), buf.begin());
    const auto *rv = state->as<ob::RealVectorStateSpace::StateType>();
    for (std::size_t i = 0; i < active_.size(); ++i)
      buf[active_[i]] = static_cast<float>(rv->values[i]);
    return Robot::Configuration(buf.data());
  }
};

class SubgroupMotionValidator : public ob::MotionValidator {
 public:
  SubgroupMotionValidator(const ob::SpaceInformationPtr &si, const VampEnv &env,
                          std::vector<int> active_indices,
                          std::vector<float> frozen_config)
      : ob::MotionValidator(si),
        env_(env),
        active_(std::move(active_indices)),
        frozen_(std::move(frozen_config)) {}

  auto checkMotion(const ob::State *s1, const ob::State *s2) const
      -> bool override {
    return vamp::planning::validate_motion<Robot, kRake, Robot::resolution>(
        expand(s1), expand(s2), env_);
  }

  auto checkMotion(const ob::State *s1, const ob::State *s2,
                   std::pair<ob::State *, double> &last_valid) const
      -> bool override {
    last_valid.first = nullptr;
    last_valid.second = 0.0;
    return checkMotion(s1, s2);
  }

 private:
  const VampEnv &env_;
  std::vector<int> active_;
  std::vector<float> frozen_;

  auto expand(const ob::State *state) const -> Robot::Configuration {
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars>
            buf{};
    std::copy(frozen_.begin(), frozen_.end(), buf.begin());
    const auto *rv = state->as<ob::RealVectorStateSpace::StateType>();
    for (std::size_t i = 0; i < active_.size(); ++i)
      buf[active_[i]] = static_cast<float>(rv->values[i]);
    return Robot::Configuration(buf.data());
  }
};

// ─── Main planner class ─────────────────────────────────────────────

struct PlanResult {
  bool solved;
  std::vector<std::vector<double>> path;
  int64_t planning_time_ns;
  double path_cost;
};

class OmplVampPlanner {
 public:
  /// Full-body constructor (24 DOF).
  OmplVampPlanner() : active_dim_(Robot::dimension), is_subgroup_(false) {
    // Build full state space with VAMP joint bounds
    Robot::Configuration lo, hi;
    std::array<float, Robot::dimension> zeros{}, ones{};
    ones.fill(1.0f);
    lo = Robot::Configuration(zeros.data());
    hi = Robot::Configuration(ones.data());
    Robot::scale_configuration(lo);
    Robot::scale_configuration(hi);

    auto lo_arr = lo.to_array();
    auto hi_arr = hi.to_array();

    auto space = std::make_shared<ob::RealVectorStateSpace>(Robot::dimension);
    ob::RealVectorBounds bounds(Robot::dimension);
    for (std::size_t i = 0; i < Robot::dimension; ++i) {
      bounds.setLow(i, std::min(lo_arr[i], hi_arr[i]));
      bounds.setHigh(i, std::max(lo_arr[i], hi_arr[i]));
    }
    space->setBounds(bounds);
    space_ = space;
  }

  /// Subgroup constructor (reduced DOF).
  OmplVampPlanner(std::vector<int> active_indices,
                  std::vector<double> frozen_config)
      : active_dim_(active_indices.size()),
        is_subgroup_(true),
        active_indices_(std::move(active_indices)) {
    frozen_config_.resize(frozen_config.size());
    for (std::size_t i = 0; i < frozen_config.size(); ++i)
      frozen_config_[i] = static_cast<float>(frozen_config[i]);

    // Build full bounds, then extract active subset
    Robot::Configuration lo, hi;
    std::array<float, Robot::dimension> zeros{}, ones{};
    ones.fill(1.0f);
    lo = Robot::Configuration(zeros.data());
    hi = Robot::Configuration(ones.data());
    Robot::scale_configuration(lo);
    Robot::scale_configuration(hi);
    auto lo_arr = lo.to_array();
    auto hi_arr = hi.to_array();

    auto space = std::make_shared<ob::RealVectorStateSpace>(active_dim_);
    ob::RealVectorBounds bounds(active_dim_);
    for (std::size_t i = 0; i < active_indices_.size(); ++i) {
      auto idx = active_indices_[i];
      bounds.setLow(i, std::min(lo_arr[idx], hi_arr[idx]));
      bounds.setHigh(i, std::max(lo_arr[idx], hi_arr[idx]));
    }
    space->setBounds(bounds);
    space_ = space;
  }

  void add_pointcloud(const std::vector<std::array<float, 3>> &points,
                      float r_min, float r_max, float point_radius) {
    std::vector<vamp::collision::Point> pts;
    pts.reserve(points.size());
    for (const auto &p : points) pts.push_back({p[0], p[1], p[2]});
    float_env_.pointclouds.emplace_back(pts, r_min, r_max, point_radius);
    sync_env();
  }

  void add_sphere(const std::array<float, 3> &center, float radius) {
    float_env_.spheres.emplace_back(vamp::collision::Sphere<float>{
        center[0], center[1], center[2], radius});
    float_env_.sort();
    sync_env();
  }

  void clear_environment() {
    float_env_ = FloatEnv{};
    env_ = VampEnv{};
  }

  auto plan(std::vector<double> start, std::vector<double> goal,
            const std::string &planner_name, double time_limit, bool simplify,
            bool interpolate) -> PlanResult {
    auto si = std::make_shared<ob::SpaceInformation>(space_);

    // Attach VAMP collision checking
    if (is_subgroup_) {
      si->setStateValidityChecker(std::make_shared<SubgroupValidityChecker>(
          si, env_, active_indices_, frozen_config_));
      si->setMotionValidator(std::make_shared<SubgroupMotionValidator>(
          si, env_, active_indices_, frozen_config_));
    } else {
      si->setStateValidityChecker(
          std::make_shared<AutolifeValidityChecker>(si, env_));
      si->setMotionValidator(
          std::make_shared<AutolifeMotionValidator>(si, env_));
    }

    og::SimpleSetup ss(si);
    ss.setPlanner(create_planner(si, planner_name));

    // Set start and goal
    ob::ScopedState<> ompl_start(space_);
    ob::ScopedState<> ompl_goal(space_);
    for (int i = 0; i < active_dim_; ++i) {
      ompl_start[i] = start[i];
      ompl_goal[i] = goal[i];
    }
    ss.setStartAndGoalStates(ompl_start, ompl_goal);

    // Solve
    auto t0 = std::chrono::steady_clock::now();
    auto status = ss.solve(time_limit);
    auto t1 = std::chrono::steady_clock::now();
    auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    PlanResult result;
    result.planning_time_ns = elapsed_ns;
    result.solved = static_cast<bool>(status);

    if (result.solved) {
      if (simplify) ss.simplifySolution();

      auto &path = ss.getSolutionPath();
      // Interpolate after simplify so the returned path has enough
      // waypoints to animate smoothly — OMPL's default uses the
      // longest valid segment fraction of the state space.
      if (interpolate) path.interpolate();
      result.path_cost = path.length();

      for (std::size_t i = 0; i < path.getStateCount(); ++i) {
        const auto *rv =
            path.getState(i)->as<ob::RealVectorStateSpace::StateType>();
        std::vector<double> config(active_dim_);
        for (int j = 0; j < active_dim_; ++j) config[j] = rv->values[j];
        result.path.push_back(std::move(config));
      }
    } else {
      result.path_cost = std::numeric_limits<double>::infinity();
    }

    return result;
  }

  auto validate(std::vector<double> config) -> bool {
    if (is_subgroup_) {
      alignas(Robot::Configuration::S::Alignment)
          std::array<float, Robot::Configuration::num_scalars>
              buf{};
      std::copy(frozen_config_.begin(), frozen_config_.end(), buf.begin());
      for (std::size_t i = 0; i < active_indices_.size(); ++i)
        buf[active_indices_[i]] = static_cast<float>(config[i]);
      auto q = Robot::Configuration(buf.data());
      return vamp::planning::validate_motion<Robot, kRake, 1>(q, q, env_);
    } else {
      alignas(Robot::Configuration::S::Alignment)
          std::array<float, Robot::Configuration::num_scalars>
              buf{};
      for (std::size_t i = 0; i < Robot::dimension; ++i)
        buf[i] = static_cast<float>(config[i]);
      auto q = Robot::Configuration(buf.data());
      return vamp::planning::validate_motion<Robot, kRake, 1>(q, q, env_);
    }
  }

  auto dimension() const -> int { return active_dim_; }

  auto lower_bounds() const -> std::vector<double> {
    auto bounds = space_->as<ob::RealVectorStateSpace>()->getBounds();
    std::vector<double> lo(active_dim_);
    for (int i = 0; i < active_dim_; ++i) lo[i] = bounds.low[i];
    return lo;
  }

  auto upper_bounds() const -> std::vector<double> {
    auto bounds = space_->as<ob::RealVectorStateSpace>()->getBounds();
    std::vector<double> hi(active_dim_);
    for (int i = 0; i < active_dim_; ++i) hi[i] = bounds.high[i];
    return hi;
  }

  auto min_max_radii() const -> std::pair<float, float> {
    return {Robot::min_radius, Robot::max_radius};
  }

 private:
  int active_dim_;
  bool is_subgroup_;
  std::vector<int> active_indices_;
  std::vector<float> frozen_config_;
  ob::StateSpacePtr space_;
  FloatEnv float_env_;
  VampEnv env_;

  void sync_env() { env_ = VampEnv(float_env_); }

  static auto create_planner(const ob::SpaceInformationPtr &si,
                             const std::string &name) -> ob::PlannerPtr {
    // RRT family
    if (name == "rrtc" || name == "rrtconnect")
      return std::make_shared<og::RRTConnect>(si);
    if (name == "rrt") return std::make_shared<og::RRT>(si);
    if (name == "rrtstar") return std::make_shared<og::RRTstar>(si);
    if (name == "informed_rrtstar")
      return std::make_shared<og::InformedRRTstar>(si);
    if (name == "rrtsharp") return std::make_shared<og::RRTsharp>(si);
    if (name == "rrtxstatic") return std::make_shared<og::RRTXstatic>(si);
    if (name == "strrtstar") return std::make_shared<og::STRRTstar>(si);
    if (name == "lbtrrt") return std::make_shared<og::LBTRRT>(si);
    if (name == "trrt") return std::make_shared<og::TRRT>(si);
    if (name == "bitrrt") return std::make_shared<og::BiTRRT>(si);
    // Informed trees (asymptotically optimal)
    if (name == "bitstar") return std::make_shared<og::BITstar>(si);
    if (name == "abitstar") return std::make_shared<og::ABITstar>(si);
    if (name == "aitstar") return std::make_shared<og::AITstar>(si);
    if (name == "eitstar") return std::make_shared<og::EITstar>(si);
    if (name == "blitstar") return std::make_shared<og::BLITstar>(si);
    // FMT
    if (name == "fmt") return std::make_shared<og::FMT>(si);
    if (name == "bfmt") return std::make_shared<og::BFMT>(si);
    // KPIECE
    if (name == "kpiece") return std::make_shared<og::KPIECE1>(si);
    if (name == "bkpiece") return std::make_shared<og::BKPIECE1>(si);
    if (name == "lbkpiece") return std::make_shared<og::LBKPIECE1>(si);
    // PRM family
    if (name == "prm") return std::make_shared<og::PRM>(si);
    if (name == "prmstar") return std::make_shared<og::PRMstar>(si);
    if (name == "lazyprm") return std::make_shared<og::LazyPRM>(si);
    if (name == "lazyprmstar") return std::make_shared<og::LazyPRMstar>(si);
    if (name == "spars") return std::make_shared<og::SPARS>(si);
    if (name == "spars2") return std::make_shared<og::SPARStwo>(si);
    // Exploration-based
    if (name == "est") return std::make_shared<og::EST>(si);
    if (name == "biest") return std::make_shared<og::BiEST>(si);
    if (name == "sbl") return std::make_shared<og::SBL>(si);
    if (name == "stride") return std::make_shared<og::STRIDE>(si);
    if (name == "pdst") return std::make_shared<og::PDST>(si);
    throw std::invalid_argument("Unknown planner: " + name);
  }
};

// ─── nanobind module ─────────────────────────────────────────────────

NB_MODULE(_ompl_vamp, m) {
  m.doc() = "OMPL + VAMP C++ planning extension for Autolife robot";

  nb::class_<PlanResult>(m, "PlanResult")
      .def_ro("solved", &PlanResult::solved)
      .def_ro("path", &PlanResult::path)
      .def_ro("planning_time_ns", &PlanResult::planning_time_ns)
      .def_ro("path_cost", &PlanResult::path_cost);

  nb::class_<OmplVampPlanner>(m, "OmplVampPlanner")
      .def(nb::init<>(), "Create a full-body planner (24 DOF).")
      .def(nb::init<std::vector<int>, std::vector<double>>(),
           "Create a subgroup planner.", nb::arg("active_indices"),
           nb::arg("frozen_config"))
      .def("add_pointcloud", &OmplVampPlanner::add_pointcloud,
           nb::arg("points"), nb::arg("r_min"), nb::arg("r_max"),
           nb::arg("point_radius"))
      .def("add_sphere", &OmplVampPlanner::add_sphere, nb::arg("center"),
           nb::arg("radius"))
      .def("clear_environment", &OmplVampPlanner::clear_environment)
      .def("plan", &OmplVampPlanner::plan, nb::arg("start"), nb::arg("goal"),
           nb::arg("planner_name") = "rrtc", nb::arg("time_limit") = 10.0,
           nb::arg("simplify") = true, nb::arg("interpolate") = true)
      .def("validate", &OmplVampPlanner::validate, nb::arg("config"))
      .def("dimension", &OmplVampPlanner::dimension)
      .def("lower_bounds", &OmplVampPlanner::lower_bounds)
      .def("upper_bounds", &OmplVampPlanner::upper_bounds)
      .def("min_max_radii", &OmplVampPlanner::min_max_radii);
}
