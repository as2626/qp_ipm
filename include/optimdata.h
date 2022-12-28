#include <Eigen/Dense>
#include <Eigen/SparseCholesky>

#include "linesearch.h"

using Eigen::MatrixXd;
using Eigen::seq;
using Eigen::VectorXd;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::ArithmeticSequence<Eigen::Index, Eigen::Index> Seq;

struct VariablesSize {
    size_t x;
    size_t s;
    size_t z;
    size_t y;
};
struct VariablesIndex {
    Seq x;
    Seq s;
    Seq z;
    Seq y;
};

struct QPProblem;

struct Variables {
    VectorXd x;
    VectorXd s;
    VectorXd z;
    VectorXd y;

    Variables(){};
    Variables(const QPProblem &ref);
    Variables(VectorXd &x, VectorXd &s, VectorXd &z, VectorXd &y);
    void update(VectorXd &xx, VectorXd &ss, VectorXd &zz, VectorXd &yy);
    double duality_gap() const;
    VariablesSize get_sizes() const;
    VariablesIndex get_indices(const VariablesSize &n) const;
    VariablesIndex get_indices() const;
    Variables(const VectorXd &vec, const VariablesSize &n);
    Variables operator+=(Variables vars) {
        x += vars.x;
        s += vars.s;
        z += vars.z;
        y += vars.y;
        return *this;
    }
    Variables operator*=(double a) {
        x *= a;
        s *= a;
        z *= a;
        y *= a;
        return *this;
    }
};
Variables operator*(double a, Variables vars);
std::ostream &operator<<(std::ostream &os, Variables const &vars);

struct QPProblem {
    // objective
    MatrixXd Q;
    VectorXd q;
    // inequality constraints
    MatrixXd G;
    VectorXd h;
    // equality constraints
    MatrixXd A;
    VectorXd b;

    size_t m;
    size_t n;
    size_t p;

    QPProblem(){};
    QPProblem(MatrixXd &QQ, VectorXd &qq, MatrixXd &GG, VectorXd &hh,
              MatrixXd &AA, VectorXd &bb);

    VectorXd objective_gradient(const Variables &vars) const;
    VectorXd inequality_residual(const Variables &vars) const;
    VectorXd equality_residual(const Variables &vars) const;
    VectorXd lagrangian(const Variables &vars) const;
};

struct KKTSystem {
    MatrixXd K;
    MatrixXd K_reg;
    VectorXd rhs_aff;
    VectorXd rhs_cc;

    KKTSystem(){};
    KKTSystem(const QPProblem &data, const Variables &vars);
    KKTSystem(const QPProblem &data, const Variables &vars, double kkt_reg);

    void update_duality(const Variables &vars, double kkt_reg);

    void set_rhs_affine(const QPProblem &data, const Variables &vars);
    void set_rhs_cc(const VectorXd &delta_aff, const Variables &vars);
    void iterative_refinement(VectorXd &guess, const VectorXd &rhs,
                              const Eigen::SimplicialLDLT<SpMat> &ldlt,
                              int refine_iters);
};

struct SolverOptions {
    double duality_tol = 1e-6;
    double resid_tol = 1e-4;
    int max_iters = 25;
    double kkt_reg = 1e-7;
    int refine_iters = 1;
};