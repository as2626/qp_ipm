#define EIGEN_RUNTIME_NO_MALLOC

#include <Eigen/Dense>
#include <iostream>

#include "ipm.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int argc, char** argv) {
    int n = 10;

    // Setup problem data
    double gamma = 0.5;  // Risk aversion parameter
    Eigen::MatrixXd A(1, n);
    Eigen::MatrixXd Q(n, n);  // Return Covariance
    Eigen::MatrixXd G(n, n);
    Eigen::VectorXd q(n);  // Expected Return
    Eigen::VectorXd b(1);
    Eigen::VectorXd h(n);

    // Generate random return covariance matrix (PSD matrix)
    Q.setRandom();
    Q = Q.transpose().eval() + Q;
    Q += n * Eigen::MatrixXd::Identity(n, n);
    Q *= gamma;

    // Generate random expected return
    q.setRandom();

    // No shorting (non-negativity constraint)
    G = -Eigen::MatrixXd::Identity(n, n);
    h.setZero(n);

    // Variables represent fraction of investment
    A.setConstant(1);
    b << 1;

    Solver solver(Q, q, G, h, A, b);
    solver.solve();

    // Check solver status
    std::cout << "Converged?" << std::endl;
    std::cout << solver.converged() << std::endl;

    // Print optimal portfolio
    std::cout << "x_opt:" << std::endl;
    std::cout << solver.vars.x << std::endl;
}
