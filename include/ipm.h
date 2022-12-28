#include <Eigen/Dense>

#include "optimdata.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

struct Solver {
    QPProblem data;
    Variables vars;
    KKTSystem kkt;

    Eigen::SimplicialLDLT<SpMat> ldlt;
    VectorXd delta_aff;
    VectorXd delta_cc;
    Variables delta;
    double step_size;

    SolverOptions options;

    Solver(MatrixXd &Q, VectorXd &q, MatrixXd &G, VectorXd &h, MatrixXd &A,
           VectorXd &b);

    bool converged();
    bool converged(const QPProblem &data, const Variables &vars,
                   double resid_tol, double duality_tol);

    void solve();
};
