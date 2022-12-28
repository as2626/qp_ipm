#include "linesearch.h"

double non_neg_step_size(const VectorXd &vec, const VectorXd &dirn) {
    VectorXd candidates = -vec.cwiseQuotient(dirn);
    for (int i = 0; i < vec.size(); ++i) {
        if (dirn[i] >= 0) {  // can step in non-negative directions until 1, and
                             // avoid division-by-zero
            candidates[i] = 1.;
        }
    }
    return fmin(1., candidates.minCoeff());
}
