This repository implements an interior-point method to solve quadratic programs of the form
$$
\begin{split}
    \underset{x}{\text{minimize}} 
    \quad & x^\top Q x + q^\top x \\
    \text{subject to} 
    \quad & Gx \preceq h, \\
    \quad & Ax = b.
\end{split}
$$
Example usage is in `main.cpp`. Implementation follows the outlined algorithm in Section 5 of the [CVXGEN](https://stanford.edu/~boyd/papers/pdf/code_gen_impl.pdf) paper. The core logic is implemented in `solve()` function of `src/ipm.cpp`.

The algorithm includes:
* Prediction (affine), centering, and correcting steps
* Permuted $LDL^\top$ factorization to solve the formulated quasi-definite KKT system, to exploit sparsity
* KKT system regularization, and iterative refinement

Usage follows as
```
git clone --recurse-submodules git@github.com:as2626/qp_ipm.git \
&& mkdir build
```
and possibly
```
cd eigen \
&& mkdir build \
&& cd build \
&& cmake .. \
&& make install
```

## Acknowledgement

N.B., much credit goes to [Govind Chari](https://github.com/govindchari/QPSolver) for a reference implementation.
