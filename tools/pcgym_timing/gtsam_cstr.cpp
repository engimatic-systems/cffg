// Native residual and forward-mode RK4 Jacobian for cstr_runtime_fixture.cstr_rk4.
#include <gtsam/nonlinear/CustomFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <unsupported/Eigen/AutoDiff>

#include <cmath>
#include <limits>

namespace py = pybind11;
using gtsam::Vector;
using gtsam::Vector2;
using gtsam::Vector4;

template <typename Scalar>
using State = Eigen::Matrix<Scalar, 4, 1>;
template <typename Scalar>
using Physical = Eigen::Matrix<Scalar, 2, 1>;

// jnp.clip and jnp.maximum split the tangent equally between tied operands.
template <typename Scalar>
Scalar clip(Scalar x, double lo, double hi) {
  if (x < lo) return Scalar(lo);
  if (x > hi) return Scalar(hi);
  if (x == lo) return (x + lo) * 0.5;
  if (x == hi) return (x + hi) * 0.5;
  return x;
}

struct Reactor {
  Vector2 lo, span;
  double uLo, uHi, rate, activation, heatTransfer, reactionHeat, step;
  int substeps, tracked;

  template <typename Scalar>
  State<Scalar> advance(const State<Scalar>& state) const {
    using std::exp;
    Physical<Scalar> x;
    for (int i = 0; i < 2; ++i) {
      Scalar raw = (state[i] + 1.) / 2. * span[i] + lo[i];
      x[i] = clip(raw, lo[i] - 2. * span[i], lo[i] + 3. * span[i]);
    }
    x[0] = clip(x[0], 0., std::numeric_limits<double>::infinity());
    Scalar rawU = (state[3] + 1.) / 2. * (uHi - uLo) + uLo;
    Scalar u = clip(rawU, uLo - 20., uHi + 20.);
    auto derivative = [&](const Physical<Scalar>& v) -> Physical<Scalar> {
      Scalar reaction = rate * exp(-activation * (1. / v[1] - 1. / 330.)) * v[0];
      return {1. - v[0] - reaction,
              350. - v[1] + reactionHeat * reaction / (1000. * .239)
                  + heatTransfer * (u - v[1]) / (1000. * .239 * 100.)};
    };

    for (int i = 0; i < substeps; ++i) {
      const auto k1 = derivative(x);
      const auto k2 = derivative(x + .5 * step * k1);
      const auto k3 = derivative(x + .5 * step * k2);
      const auto k4 = derivative(x + step * k3);
      x += step / 6. * (k1 + 2. * k2 + 2. * k3 + k4);
    }
    State<Scalar> out = state;
    for (int i = 0; i < 2; ++i) out[i] = 2. * (x[i] - lo[i]) / span[i] - 1.;
    out[tracked] += state[2];
    return out;
  }

  std::shared_ptr<gtsam::CustomFactor> factor(
      gtsam::Key i, const gtsam::SharedNoiseModel& noise) const {
    auto residual = [reactor = *this, i](const gtsam::CustomFactor&,
                      const gtsam::Values& values, gtsam::OptionalJacobianVector H) -> Vector {
      const Vector4 state = values.at<Vector>(i);
      const auto& next = values.at<Vector>(i + 1);
      if (!H) return next - reactor.advance(state);

      using Dual = Eigen::AutoDiffScalar<Vector4>;
      State<Dual> point;
      for (int j = 0; j < 4; ++j) point[j] = Dual(state[j], Vector4::Unit(j));
      const auto out = reactor.advance(point);
      Vector4 predicted;
      gtsam::Matrix4 jacobian;
      for (int row = 0; row < 4; ++row) {
        predicted[row] = out[row].value();
        jacobian.row(row) = -out[row].derivatives().transpose();
      }
      H->get()[0] = jacobian;
      H->get()[1] = gtsam::Matrix4::Identity();
      return next - predicted;
    };
    return std::make_shared<gtsam::CustomFactor>(noise, gtsam::KeyVector{i, i + 1}, residual);
  }
};

PYBIND11_MODULE(_gtsam_cstr, module) {
  py::class_<Reactor>(module, "Reactor")
      .def(py::init([](const Vector4& theta, const py::dict& cfg) {
        const auto lo = cfg["x_lo"].cast<Vector2>();
        const auto span = (cfg["x_hi"].cast<Vector2>() - lo).eval();
        int substeps = cfg["substeps"].cast<int>();
        const auto tracked = cfg["tracked"].cast<py::sequence>();
        if (cfg["nd"].cast<int>() != 1 || tracked.size() != 1 ||
            tracked[0].cast<int>() < 0 || tracked[0].cast<int>() > 1 || substeps <= 0)
          throw std::invalid_argument("Expected one tracked CSTR state and positive substeps");
        return Reactor{lo, span, cfg["u_lo"].cast<double>(), cfg["u_hi"].cast<double>(),
                       std::exp(theta[0]), theta[1] * 1e4, std::exp(theta[2]), std::exp(theta[3]),
                       cfg["dt"].cast<double>() / substeps, substeps, tracked[0].cast<int>()};
      }))
      .def("factor", &Reactor::factor);
}
