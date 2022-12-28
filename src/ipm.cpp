#include "ipm.h"

Solver::Solver(MatrixXd &Q, VectorXd &q, MatrixXd &G, VectorXd &h, MatrixXd &A,
               VectorXd &b) {
    data = QPProblem(Q, q, G, h, A, b);
}

bool Solver::converged() {
    return converged(data, vars, options.resid_tol, options.duality_tol);
}
bool Solver::converged(const QPProblem &data, const Variables &vars,
                       double resid_tol, double duality_tol) {
    bool converged = true;
    if (data.G.size() > 0) {
        converged &= (data.inequality_residual(vars).norm() < resid_tol);
        converged &= (vars.duality_gap() < duality_tol);
    }
    if (data.A.size() > 0)
        converged &= (data.equality_residual(vars).norm() < resid_tol);
    return converged;
}

void Solver::solve() {
    // Initialization
    vars = Variables(data);
    kkt = KKTSystem(data, vars, options.kkt_reg);

    auto i = 0;

    // 1. Converged?
    while (!converged(data, vars, options.resid_tol, options.duality_tol) &&
           (i < options.max_iters)) {
        ++i;
        // Form KKT for 2. and 3.
        kkt.update_duality(vars, options.kkt_reg);
        ldlt.compute(kkt.K_reg.sparseView());

        // 2. Affine scaling directions
        kkt.set_rhs_affine(data, vars);
        delta_aff = std::move(ldlt.solve(kkt.rhs_aff));
        kkt.iterative_refinement(delta_aff, kkt.rhs_aff, ldlt,
                                 options.refine_iters);

        // 3. Centering+corrector directions
        kkt.set_rhs_cc(delta_aff, vars);
        delta_cc = std::move(ldlt.solve(kkt.rhs_cc));
        kkt.iterative_refinement(delta_cc, kkt.rhs_cc, ldlt,
                                 options.refine_iters);

        // 4. Combine updates and choose step size.
        delta = Variables(delta_aff + delta_cc, vars.get_sizes());
        step_size = 0.99 * fmin(non_neg_step_size(vars.s, delta.s),
                                non_neg_step_size(vars.z, delta.z));
        step_size = fmin(1., step_size);

        // 5. Update primal and dual variables.
        vars += step_size * delta;
    }
}
