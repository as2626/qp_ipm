#include "optimdata.h"

Variables::Variables(VectorXd &x, VectorXd &s, VectorXd &z, VectorXd &y) {
    update(x, s, z, y);
}
Variables::Variables(const QPProblem &data) {
    auto n_x = data.n;
    auto n_z = data.p;
    auto n_y = data.m;
    auto N = n_x + n_z + n_y;

    auto idx_x = seq(0, n_x - 1);
    auto idx_z = seq(n_x, n_x + n_z - 1);
    auto idx_y = seq(n_x + n_z, N - 1);

    MatrixXd mat = MatrixXd::Zero(N, N);
    mat(idx_x, idx_x) = data.Q;

    if (data.G.size() > 0) {
        mat(idx_x, idx_z) = data.G.transpose();
        mat(idx_z, idx_x) = data.G;
        mat(idx_z, idx_z) = -MatrixXd::Identity(n_z, n_z);
    }
    if (data.A.size() > 0) {
        mat(idx_x, idx_y) = data.A.transpose();
        mat(idx_y, idx_x) = data.A;
    }

    VectorXd rhs(N);
    rhs(idx_x) = -data.q;
    rhs(idx_z) = data.h;
    rhs(idx_y) = data.b;

    Eigen::SimplicialLDLT<SpMat> ldlt;
    ldlt.compute(mat.sparseView());

    auto solution = ldlt.solve(rhs);
    x = std::move(solution(idx_x));
    y = std::move(solution(idx_y));

    if (n_z > 0) {
        auto z_tmp = solution(idx_z).array();

        double alpha_p = z_tmp.maxCoeff();
        double add_to_s = alpha_p < 0 ? 0 : 1 + alpha_p;
        s = std::move((-z_tmp + add_to_s).matrix());

        double alpha_d = -(z_tmp.minCoeff());
        double add_to_z = alpha_d < 0 ? 0 : 1 + alpha_d;
        z = std::move((z_tmp + add_to_z).matrix());
    }
}
Variables::Variables(const VectorXd &vec, const VariablesSize &n) {
    auto idx = get_indices(n);
    x = vec(idx.x);
    s = vec(idx.s);
    z = vec(idx.z);
    y = vec(idx.y);
}

void Variables::update(VectorXd &xx, VectorXd &ss, VectorXd &zz, VectorXd &yy) {
    x = std::move(xx);
    s = std::move(ss);
    z = std::move(zz);
    y = std::move(yy);
}
double Variables::duality_gap() const { return s.dot(z) / s.size(); }
VariablesSize Variables::get_sizes() const {
    const VariablesSize n = {(size_t)x.size(), (size_t)s.size(),
                             (size_t)z.size(), (size_t)y.size()};
    return n;
}
VariablesIndex Variables::get_indices(const VariablesSize &n) const {
    size_t N = n.x + n.s + n.y + n.z;
    Seq xx = seq(0, n.x - 1);
    Seq ss = seq(n.x, n.x + n.s - 1);
    Seq zz = seq(n.x + n.s, n.x + n.s + n.z - 1);
    Seq yy = seq(n.x + n.s + n.z, N - 1);

    const VariablesIndex idx = {xx, ss, zz, yy};

    return idx;
}
VariablesIndex Variables::get_indices() const {
    return get_indices(get_sizes());
}
Variables operator*(double a, Variables vars) { return vars *= a; }
std::ostream &operator<<(std::ostream &os, Variables const &vars) {
    return os << "x\n"
              << vars.x << std::endl
              << "s\n"
              << vars.s << std::endl
              << "y\n"
              << vars.y << std::endl
              << "z\n"
              << vars.z;
}

QPProblem::QPProblem(MatrixXd &QQ, VectorXd &qq, MatrixXd &GG, VectorXd &hh,
                     MatrixXd &AA, VectorXd &bb) {
    Q = std::move(QQ);
    q = std::move(qq);
    G = std::move(GG);
    h = std::move(hh);
    A = std::move(AA);
    b = std::move(bb);

    m = A.rows();
    n = Q.rows();
    p = G.rows();
}

VectorXd QPProblem::objective_gradient(const Variables &vars) const {
    return Q.sparseView() * vars.x + q;
}
VectorXd QPProblem::inequality_residual(const Variables &vars) const {
    return G.sparseView() * vars.x + vars.s - h;
}
VectorXd QPProblem::equality_residual(const Variables &vars) const {
    return A.sparseView() * vars.x - b;
}
VectorXd QPProblem::lagrangian(const Variables &vars) const {
    VectorXd lag = objective_gradient(vars);
    if (A.size() > 0) {
        lag += A.transpose() * vars.y;
    }
    if (G.size() > 0) {
        lag += G.transpose() * vars.z;
    }
    return lag;
}

KKTSystem::KKTSystem(const QPProblem &data, const Variables &vars) {
    auto n = vars.get_sizes();
    auto N = n.x + n.s + n.z + n.y;
    auto idx = vars.get_indices();

    K = MatrixXd::Zero(N, N);
    K(idx.x, idx.x) = data.Q;
    if (data.G.size() > 0) {
        K(idx.x, idx.z) = data.G.transpose();
        K(idx.z, idx.x) = data.G;
        K(idx.z, idx.s) = MatrixXd::Identity(n.z, n.z);
        K(idx.s, idx.z) = MatrixXd::Identity(n.s, n.s);
    }
    if (data.A.size() > 0) {
        K(idx.x, idx.y) = data.A.transpose();
        K(idx.y, idx.x) = data.A;
    }

    rhs_aff = VectorXd(N);
    rhs_cc = VectorXd::Zero(N);
}
KKTSystem::KKTSystem(const QPProblem &data, const Variables &vars,
                     double kkt_reg)
    : KKTSystem(data, vars) {
    auto top_left = data.n + data.p;
    auto bot_right = data.m + data.p;
    auto idx_top_left = seq(0, top_left - 1);
    auto idx_bot_right = seq(top_left, top_left + bot_right - 1);

    MatrixXd reg_matrix =
        MatrixXd::Zero(top_left + bot_right, top_left + bot_right);
    reg_matrix(idx_top_left, idx_top_left) =
        kkt_reg * MatrixXd::Identity(top_left, top_left);
    reg_matrix(idx_bot_right, idx_bot_right) =
        -kkt_reg * MatrixXd::Identity(bot_right, bot_right);

    K_reg = K + reg_matrix;
}

void KKTSystem::update_duality(const Variables &vars, double kkt_reg) {
    if (vars.s.size() == 0) return;

    auto n_x = vars.x.size();
    auto n_s = vars.s.size();
    auto idx_s = seq(n_x, n_x + n_s - 1);
    auto for_update = vars.z.cwiseQuotient(vars.s);

    K(idx_s, idx_s) = for_update.asDiagonal();
    K_reg(idx_s, idx_s) =
        (for_update + kkt_reg * VectorXd::Ones(n_s)).asDiagonal();
}

void KKTSystem::set_rhs_affine(const QPProblem &data, const Variables &vars) {
    auto n = vars.get_sizes();
    auto idx = vars.get_indices();
    rhs_aff(idx.x) = -data.lagrangian(vars);

    if (n.s > 0) {
        rhs_aff(idx.s) = -vars.z;
        rhs_aff(idx.z) = -data.inequality_residual(vars);
    }

    if (n.y > 0) {
        rhs_aff(idx.y) = -data.equality_residual(vars);
    }
}

void KKTSystem::set_rhs_cc(const VectorXd &delta_aff, const Variables &vars) {
    auto n = vars.get_sizes();
    auto idx = vars.get_indices();

    double sigma = 0;
    double mu = 0;

    auto s = vars.s;
    auto z = vars.z;
    auto s_aff = delta_aff(idx.s);
    auto z_aff = delta_aff(idx.z);
    if (n.z > 0) {
        mu = vars.duality_gap();
        sigma = std::pow(sigma, 3);
    }

    rhs_cc(idx.s) = (sigma * mu) + (-s_aff.cwiseProduct(z_aff)).array();
    rhs_cc(idx.s) = rhs_cc(idx.s).cwiseQuotient(s);
}

void KKTSystem::iterative_refinement(VectorXd &guess, const VectorXd &rhs,
                                     const Eigen::SimplicialLDLT<SpMat> &ldlt,
                                     int refine_iters) {
    for (int i = 0; i < refine_iters; ++i) {
        VectorXd resid = rhs - K * guess;
        guess += ldlt.solve(resid.sparseView());
    }
}