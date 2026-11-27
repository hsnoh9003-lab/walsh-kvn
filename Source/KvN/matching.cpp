#include <cstddef>
#ifdef _OPENMP
#include <omp.h>
#endif

// Export the same C ABI on both platforms; Windows requires explicit DLL symbol export.
#ifdef _WIN32
#define KVN_EXPORT extern "C" __declspec(dllexport)
#else
#define KVN_EXPORT extern "C"
#endif

// Configure the kernel's OpenMP thread count through an exported symbol, without relying on transitive runtime exports.
KVN_EXPORT void matching_set_threads(int threads) {
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif
}

// The state is a contiguous [z][r][k] array of double-precision complex amplitudes shared with NumPy.
namespace {
struct Complex {
    double re;
    double im;
};

// Apply a real two-state rotation equivalent to exp(-i*angle*Y)
// in the matched subspace. The operation preserves the combined
// probability of the two amplitudes.
inline void real_rotation(Complex& a, Complex& b, double c, double s) {
    const Complex left = a;
    const Complex right = b;
    a = {c*left.re-s*right.re,c*left.im-s*right.im};
    b = {s*left.re+c*right.re,s*left.im+c*right.im};
}

// Apply a complex two-state rotation equivalent to exp(-i*angle*X).
// This kernel handles Fourier-mode couplings with imaginary
// off-diagonal matrix elements.
inline void imaginary_rotation(Complex& a, Complex& b, double c, double s) {
    const Complex left = a;
    const Complex right = b;
    a = {c*left.re+s*right.im,c*left.im-s*right.re};
    b = {c*right.re+s*left.im,c*right.im-s*left.re};
}

// Apply radial or vertical transport to a matched pair of spatial cells.
// The constant harmonic couples equal Fourier modes, whereas cosine
// and sine harmonics couple modes separated by two indices.
inline void rz_block(Complex* low, Complex* high, const double* angle, int nt,
                     int harmonic, int kp) {
    const double c = angle[0];
    const double s = angle[1];
    if (harmonic == 0) {
        #pragma omp simd
        for (int k = 0; k < nt; ++k) {
            real_rotation(low[k],high[k],c,s);
        }
    }
    // Cosine harmonic: couple k and k+2 across the two spatial cells
    // using disjoint mode pairs selected by kp.
    else if (harmonic == 1) {
        // cos(2θ) coupling: X(k,k+2) tensor Y on the transport pair.
        for (int k = 2*kp; k+3 < nt; k += 4) {
            real_rotation(low[k],high[k+2],c,s);
            real_rotation(low[k+2],high[k],c,s);
            real_rotation(low[k+1],high[k+3],c,s);
            real_rotation(low[k+3],high[k+1],c,s);
        }
    }
    // Sine harmonic: use the corresponding complex coupling with
    // alternating signs fixed by the Fourier-space operator.
    else {
        // sin(2θ) coupling: -Y(k,k+2) tensor Y on the transport pair.
        for (int k = 2*kp; k+3 < nt; k += 4) {
            imaginary_rotation(low[k],high[k+2],c,s);
            imaginary_rotation(low[k+2],high[k],c,-s);
            imaginary_rotation(low[k+1],high[k+3],c,s);
            imaginary_rotation(low[k+3],high[k+1],c,-s);
        }
    }
}

// Apply angular transport within one fixed (r,z) cell.
// The constant harmonic contributes a mode-dependent diagonal phase;
// cosine and sine harmonics couple disjoint k-to-k+2 mode pairs.
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

// Traverse 2*layers-1 stages forward and backward; the stored angle table already contains the half/full Strang factors.
KVN_EXPORT void matching_step(Complex* state, const int* sequence, const double* angles,
                               int nr, int nt, int nz, int layers, int table_stride) {
    const auto row_stride = static_cast<std::size_t>(nt);
    const auto plane_stride = row_stride*nr;
    // Parallelize only disjoint matching pairs within a stage; the implicit loop barrier precedes the next stage.
    #pragma omp parallel
    {
        // Execute a symmetric Strang sequence: visit all layers forward
        // and then all except the central layer in reverse order.
        // The angle tables already contain the half-step and full-step factors.
        for (int stage = 0; stage < 2*layers-1; ++stage) {
            const int layer = stage < layers ? stage : 2*layers-2-stage;
            const int* desc = sequence + 5*layer;
            const int axis = desc[0];
            const int distance = desc[1];
            const int parity = desc[2];
            const int harmonic = desc[3];
            const int kp = desc[4];
            const auto table = angles + static_cast<std::size_t>(layer)*table_stride;
            // Radial matching: rotate pairs separated by the requested stencil
            // distance. The parity and offset loops partition the valid pairs
            // into disjoint sets without connecting opposite domain boundaries.
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
            }
            // Angular-Fourier matching: keep r and z fixed and apply the
            // corresponding diagonal phase or k-to-k+2 rotation within each row.
            else if (axis == 1) {
                #pragma omp for collapse(2) schedule(static)
                for (int z = 0; z < nz; ++z) {
                    for (int r = 0; r < nr; ++r) {
                        auto row = state+z*plane_stride+r*row_stride;
                        theta_block(row,table+2*z*nt,nt,harmonic,kp);
                    }
                }
            }
            // Vertical matching: use the same pair rotations as radial transport,
            // but move between z planes rather than adjacent radial rows.
            else {
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
