#include <cstddef>

namespace {
struct Complex {
    double re;
    double im;
};

inline void real_rotation(Complex& a, Complex& b, double c, double s) {
    const Complex left = a;
    const Complex right = b;
    a = {c*left.re-s*right.re,c*left.im-s*right.im};
    b = {s*left.re+c*right.re,s*left.im+c*right.im};
}

inline void imaginary_rotation(Complex& a, Complex& b, double c, double s) {
    const Complex left = a;
    const Complex right = b;
    a = {c*left.re+s*right.im,c*left.im-s*right.re};
    b = {c*right.re+s*left.im,c*right.im-s*left.re};
}

inline void rz_block(Complex* low, Complex* high, const double* angle, int nt,
                     int harmonic, int kp) {
    const double c = angle[0];
    const double s = angle[1];
    if (harmonic == 0) {
        #pragma omp simd
        for (int k = 0; k < nt; ++k) {
            real_rotation(low[k],high[k],c,s);
        }
    } else if (harmonic == 1) {
        // cos(2θ) coupling: X(k,k+2) tensor Y on the transport pair.
        for (int k = 2*kp; k+3 < nt; k += 4) {
            real_rotation(low[k],high[k+2],c,s);
            real_rotation(low[k+2],high[k],c,s);
            real_rotation(low[k+1],high[k+3],c,s);
            real_rotation(low[k+3],high[k+1],c,s);
        }
    } else {
        // sin(2θ) coupling: -Y(k,k+2) tensor Y on the transport pair.
        for (int k = 2*kp; k+3 < nt; k += 4) {
            imaginary_rotation(low[k],high[k+2],c,s);
            imaginary_rotation(low[k+2],high[k],c,-s);
            imaginary_rotation(low[k+1],high[k+3],c,s);
            imaginary_rotation(low[k+3],high[k+1],c,-s);
        }
    }
}

inline void theta_block(Complex* row, const double* angles, int nt, int harmonic, int kp) {
    if (harmonic == 0) {
        #pragma omp simd
        for (int k = 0; k < nt; ++k) {
            const auto a = row[k];
            const auto c = angles[2*k];
            const auto s = angles[2*k+1];
            row[k] = {c*a.re+s*a.im,c*a.im-s*a.re};
        }
    } else if (harmonic == 1) {
        for (int k = 2*kp; k+3 < nt; k += 4) {
            imaginary_rotation(row[k],row[k+2],angles[2*k],angles[2*k+1]);
            imaginary_rotation(row[k+1],row[k+3],angles[2*k+2],angles[2*k+3]);
        }
    } else {
        for (int k = 2*kp; k+3 < nt; k += 4) {
            real_rotation(row[k],row[k+2],angles[2*k],-angles[2*k+1]);
            real_rotation(row[k+1],row[k+3],angles[2*k+2],-angles[2*k+3]);
        }
    }
}
}

extern "C" void matching_step(Complex* state, const int* sequence, const double* angles,
                               int nr, int nt, int nz, int layers, int table_stride) {
    const auto row_stride = static_cast<std::size_t>(nt);
    const auto plane_stride = row_stride*nr;
    #pragma omp parallel
    {
        for (int stage = 0; stage < 2*layers-1; ++stage) {
            const int layer = stage < layers ? stage : 2*layers-2-stage;
            const int* desc = sequence + 5*layer;
            const int axis = desc[0];
            const int distance = desc[1];
            const int parity = desc[2];
            const int harmonic = desc[3];
            const int kp = desc[4];
            const auto table = angles + static_cast<std::size_t>(layer)*table_stride;
            if (axis == 0) {
                #pragma omp for collapse(2) schedule(static)
                for (int z = 0; z < nz; ++z) {
                    for (int r0 = parity*distance; r0 < nr-distance; r0 += 2*distance) {
                        for (int off = 0; off < distance && r0+off < nr-distance; ++off) {
                            const int r = r0+off;
                            auto low = state+z*plane_stride+r*row_stride;
                            rz_block(low,low+distance*row_stride,table+2*(z*nr+r),nt,harmonic,kp);
                        }
                    }
                }
            } else if (axis == 1) {
                #pragma omp for collapse(2) schedule(static)
                for (int z = 0; z < nz; ++z) {
                    for (int r = 0; r < nr; ++r) {
                        auto row = state+z*plane_stride+r*row_stride;
                        theta_block(row,table+2*z*nt,nt,harmonic,kp);
                    }
                }
            } else {
                #pragma omp for collapse(2) schedule(static)
                for (int z0 = parity*distance; z0 < nz-distance; z0 += 2*distance) {
                    for (int r = 0; r < nr; ++r) {
                        for (int off = 0; off < distance && z0+off < nz-distance; ++off) {
                            const int z = z0+off;
                            auto low = state+z*plane_stride+r*row_stride;
                            rz_block(low,low+distance*plane_stride,table+2*(z*nr+r),nt,harmonic,kp);
                        }
                    }
                }
            }
        }
    }
}
