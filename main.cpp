#include <iostream>

#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

struct Newton {
    Newton(const Eigen::VectorXd &x) : _x(x), _a(x.size()) {}

    void Interpolate(const Eigen::VectorXd &y);

    double operator()(double x) const;

private:
    Eigen::VectorXd _x;    // nodes
    Eigen::VectorXd _a;    // coefficients
};

// Compute the coefficients in the Newton basis.
void Newton::Interpolate(const Eigen::VectorXd &y) {
//    Can be in O(n^2) since the matrix is triangular.
//  But i don't know if I chose the right Eigen function for this.
    int n = y.size();
    MatrixXd LHS(n, n);


    for (int i = 0; i < n; ++i) {
        LHS(i, 0) = 1;
    }


    for (int i = 1; i < n; ++i) {
        for (int j = 1; j < i + 1; ++j) {
            LHS(i, j) = LHS(i, j - 1) * (_x[i] - _x[j - 1]);
        }
    }

    _a = LHS.partialPivLu().solve(y);

}

// Evaluate the interpolant at x.
double Newton::operator()(double x) const {
    long n = _a.size();
    VectorXd b(n);
    b(n - 1) = _a(n - 1);

    for (int i = n - 2; i >= 0; --i) {
        b(i) = b(i + 1) * (x - _x[i]) + _a[i];
    }


    return b(0);
}

struct Lagrange {
    Lagrange(const Eigen::VectorXd &x);

    void Interpolate(const Eigen::VectorXd &y) { _y = y; }

    double operator()(double x) const;

private:
    Eigen::VectorXd _x;    // nodes
    Eigen::VectorXd _l;    // weights
    Eigen::VectorXd _y;    // coefficients
};

// Compute the weights l for given nodes x.
Lagrange::Lagrange(const Eigen::VectorXd &x) : _x(x), _l(x.size()), _y(x.size()) {
    int n = x.size();
    for (int j = 0; j < n; ++j) {
        double prod = 1;
        for (int i = 0; i < n; ++i) {
            if (i != j) {
                prod *= 1.0 / (_x[i] - x[j]);
            }
        }
        _l[j] = prod;
    }

}

// Evaluate the interpolant at x.
double Lagrange::operator()(double x) const {
//    Complexity: O(n^2)
    int n = _x.size();
    Eigen::VectorXd w(n);

    for (int i = 0; i < n; ++i) {
        double prod = 1;
        for (int j = 0; j < n; ++j) {
            prod *= (x - _x[j]);
        }
        w(i) = prod;
    }

    Eigen::VectorXd L(n);

    for (int i = 0; i < n; ++i) {
        L(i) = w(i) * _l[i] / (x - _x[i]);
    }


    double result = 0;

    for (int k = 0; k < n; ++k) {
        result += _y[k] * L(k);
    }


    return result;
}

// Runge function
Eigen::VectorXd r(const Eigen::VectorXd &x) {
    return (1.0 / (1.0 + 25.0 * x.array() * x.array())).matrix();
}

int main() {
    int n = 5;
    Eigen::VectorXd x;
    x.setLinSpaced(5, -1.0, 1.0);
    Eigen::VectorXd y = r(x);

    Newton p(x);
    p.Interpolate(y); // correct result: p._a = [0.0384615, 0.198939, 1.5252, -3.31565, 3.31565]

    Lagrange q(x);    // correct result: p._l = [0.666667, -2.66667, 4, -2.66667, 0.666667]
    q.Interpolate(y);

    // Compute difference of p and q.
    int m = 22;
    double offset = 0.08333333333;
    x.setLinSpaced(m, -1.0 + offset, 1.0 - offset);
    double norm2 = .0;
    for (int i = 0; i < m; ++i) {
        double d = p(x(i)) - q(x(i));
        norm2 += d * d;
    }

    // By uniquenss of the interpolation polynomial, we expect p = q.
    std::cout << "This number should be close to zero: " << norm2 << std::endl;

    return 0;
}

