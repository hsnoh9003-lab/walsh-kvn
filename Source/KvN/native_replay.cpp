#include <complex>
#include <cstddef>
#include <cmath>

// In-place elementary gate replay, little-endian qubits. No gate fusion,
// phase alignment, probability projection, or clean-work substitution.
extern "C" void replay_native(std::complex<double>* s, std::size_t size,
    const int* gate, const int* q0, const int* q1, const double* angle, int count) {
    const double invsqrt2=1/std::sqrt(2.);
    for(int g=0;g<count;++g){
        const std::size_t bit=std::size_t(1)<<q0[g];
        const int op=gate[g];
        if(op==0||op==1){ // H or X
            for(std::size_t base=0;base<size;base+=2*bit)
                for(std::size_t j=base;j<base+bit;++j){
                    auto a=s[j],b=s[j+bit];
                    if(op==0){s[j]=(a+b)*invsqrt2;s[j+bit]=(a-b)*invsqrt2;}
                    else{s[j]=b;s[j+bit]=a;}
                }
        }else if(op==2){ // CX: q0 control, q1 target
            const std::size_t target=std::size_t(1)<<q1[g];
            for(std::size_t base=0;base<size;base+=2*target)
                for(std::size_t j=base;j<base+target;++j)
                    if(j&bit)std::swap(s[j],s[j+target]);
        }else{
            const auto low=op==3?std::polar(1.,-.5*angle[g]):std::complex<double>(1.,0.);
            const auto high=std::polar(1.,op==3?.5*angle[g]:angle[g]);
            for(std::size_t base=0;base<size;base+=2*bit)
                for(std::size_t j=base;j<base+bit;++j){s[j]*=low;s[j+bit]*=high;}
        }
    }
}
