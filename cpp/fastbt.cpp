// Fast cross-sectional backtest kernels exposed to Python with pybind11.
//
// long_short(scores, returns, quantile)
//     Each day, go long the top `quantile` fraction of stocks by score and short the
//     bottom fraction, equal-weighted, +0.5 / -0.5 gross per side. Membership is
//     decided from the scores alone (a stock with a missing *future* return is still
//     held and earns 0), so the backtest never peeks at which stocks have future data.
//     Returns (pnl[T], turnover[T]).
//
// rank_ic(scores, returns)
//     Daily Spearman rank correlation between scores and realized returns, using
//     stocks where both are finite. Returns ic[T] (NaN when fewer than 10 stocks).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

// Average ranks (1-based) with ties sharing the mean rank, matching scipy.stats.rankdata.
void average_ranks(const std::vector<double>& x, std::vector<double>& out) {
    const size_t n = x.size();
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return x[a] < x[b]; });
    out.assign(n, 0.0);
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j + 1 < n && x[idx[j + 1]] == x[idx[i]]) ++j;
        const double r = 0.5 * (static_cast<double>(i) + static_cast<double>(j)) + 1.0;
        for (size_t k = i; k <= j; ++k) out[idx[k]] = r;
        i = j + 1;
    }
}

double pearson(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t n = a.size();
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        sab += da * db; saa += da * da; sbb += db * db;
    }
    if (saa <= 0 || sbb <= 0) return std::numeric_limits<double>::quiet_NaN();
    return sab / std::sqrt(saa * sbb);
}

void check_shapes(const py::buffer_info& s, const py::buffer_info& r) {
    if (s.ndim != 2 || r.ndim != 2 || s.shape[0] != r.shape[0] || s.shape[1] != r.shape[1])
        throw std::invalid_argument("scores and returns must be 2-D arrays of the same shape");
}

}  // namespace

py::tuple long_short(Array scores, Array returns, double quantile) {
    if (!(quantile > 0 && quantile <= 0.5)) throw std::invalid_argument("quantile must be in (0, 0.5]");
    auto s = scores.request(), r = returns.request();
    check_shapes(s, r);
    const ssize_t T = s.shape[0], N = s.shape[1];
    const double* S = static_cast<double*>(s.ptr);
    const double* R = static_cast<double*>(r.ptr);

    Array pnl_arr(T), turn_arr(T);
    double* pnl = static_cast<double*>(pnl_arr.request().ptr);
    double* turn = static_cast<double*>(turn_arr.request().ptr);

    std::vector<double> w_prev(N, 0.0), w(N, 0.0);
    std::vector<ssize_t> valid;
    valid.reserve(N);
    for (ssize_t t = 0; t < T; ++t) {
        std::fill(w.begin(), w.end(), 0.0);
        valid.clear();
        for (ssize_t i = 0; i < N; ++i)
            if (std::isfinite(S[t * N + i])) valid.push_back(i);
        const ssize_t m = static_cast<ssize_t>(valid.size());
        const ssize_t k = static_cast<ssize_t>(std::floor(quantile * m));
        if (k >= 1) {
            std::stable_sort(valid.begin(), valid.end(),
                      [&](ssize_t a, ssize_t b) { return S[t * N + a] < S[t * N + b]; });
            for (ssize_t j = 0; j < k; ++j) {
                w[valid[j]] = -0.5 / k;          // bottom: short
                w[valid[m - 1 - j]] = 0.5 / k;   // top: long
            }
        }
        double p = 0.0, to = 0.0;
        for (ssize_t i = 0; i < N; ++i) {
            const double ret = R[t * N + i];
            if (w[i] != 0.0 && std::isfinite(ret)) p += w[i] * ret;
            to += std::fabs(w[i] - w_prev[i]);
        }
        pnl[t] = (k >= 1) ? p : std::numeric_limits<double>::quiet_NaN();
        turn[t] = to;
        w_prev.swap(w);
    }
    return py::make_tuple(pnl_arr, turn_arr);
}

Array rank_ic(Array scores, Array returns) {
    auto s = scores.request(), r = returns.request();
    check_shapes(s, r);
    const ssize_t T = s.shape[0], N = s.shape[1];
    const double* S = static_cast<double*>(s.ptr);
    const double* R = static_cast<double*>(r.ptr);
    Array out_arr(T);
    double* out = static_cast<double*>(out_arr.request().ptr);
    std::vector<double> a, b, ra, rb;
    for (ssize_t t = 0; t < T; ++t) {
        a.clear(); b.clear();
        for (ssize_t i = 0; i < N; ++i) {
            const double x = S[t * N + i], y = R[t * N + i];
            if (std::isfinite(x) && std::isfinite(y)) { a.push_back(x); b.push_back(y); }
        }
        if (a.size() < 10) { out[t] = std::numeric_limits<double>::quiet_NaN(); continue; }
        average_ranks(a, ra);
        average_ranks(b, rb);
        out[t] = pearson(ra, rb);
    }
    return out_arr;
}

PYBIND11_MODULE(_fastbt, m) {
    m.doc() = "C++ kernels for cross-sectional long-short backtests";
    m.def("long_short", &long_short, py::arg("scores"), py::arg("returns"), py::arg("quantile") = 0.1);
    m.def("rank_ic", &rank_ic, py::arg("scores"), py::arg("returns"));
}
