// This file contains various functions to compute the integral
//
//      int_0^1 t^j exp(-2 pi alpha t) dt
//
// We have a few different methods to compute this depeding on
// the range of input. The simplest method is perhaps to use an
// exact expression for the antiderivative of the integrand,
// but for certain ranges of input there can be some bad cancellation
// in the sum that comes up, so this does not always work well.
//
// Currently we also employ a method which uses the taylor
// series expansion for exp(-2 pi alpha t) and then integrates
// term-by-term and a continued fraction based method which
// was taken from lcalc.
//


#include "theta_sums.h"
#include <iostream>

#include "precomputed_tables.h"

using namespace std;

inline double real(double x) {
    return x;
}
inline double imag(double x) {
    return 0;
}
inline double norm(double x) {
    return x * x;
}
template<typename T> Complex H(int j, T alpha, Double epsilon);
template Complex H<Complex>(int j, Complex alpha, Double epsilon);
template Complex H<double>(int j, double alpha, Double epsilon);

template<typename T> Complex H_method1(int j, T alpha);
inline Complex H_method1_I(int j, Double alpha);
template<typename T> inline T H_method2(int j, T alpha, Double epsilon);
template<typename T> T H_method4(int j, T alpha, Double epsilon);

template <typename T> Complex H_method1(int j, T alpha) {
    // In this case we compute an "exact" value using the antiderivative of the integrand.
    // 

    if(real(alpha) == 0) {
        return H_method1_I(j, imag(alpha));
    }

    T S = (T)0.0;
    Double j_factorial = factorial(j);
    T alpha_power = (T)1;
    for(int v = 0; v < j + 1; v++) {
        if(v > 0) {
    //        v_factorial *= v;
            alpha_power *= alpha;
        }
        T z = alpha_power * two_pi_over_factorial_power(v);  //two_pi_alpha_power/factorial(v);
        S = S + z;
    }
    S = S * EXP(-2 * PI * alpha);
    S = (Double)1.0 - S;
    alpha_power *= alpha;
    //S = S * j_factorial/(alpha_power * two_pi_power(j+1));
    S = S * j_factorial;
    S = S/two_pi_power(j+1);
    S = S/alpha_power;

    return S;
}

inline Complex H_method1_I(int j, Double alpha) {
    // In this case we compute an "exact" value using the antiderivative of the integrand.
    // 
    // Specialized for imaginary alpha. Here alpha has an implicit I multiplying it.
    Complex S = 0.0;
    Double j_factorial = factorial(j);
    Double alpha_power = 1.0;
    for(int v = 0; v < j + 1; v++) {
        if(v > 0) {
    //        v_factorial *= v;
            alpha_power *= alpha;
        }
        Double z = alpha_power * two_pi_over_factorial_power(v);  //two_pi_alpha_power/factorial(v);
        S = S + I_power(v) * z;
    }
    //S = S * exp(-2 * PI * alpha);
    S = S * Complex( cos(-2 * PI * alpha), sin(-2 * PI * alpha) );
    S = (Double)1.0 - S;
    alpha_power *= alpha;
    S = S * j_factorial/(I_power(j + 1) * alpha_power * two_pi_power(j+1));


    return S;
}

template<typename T> inline T H_method2(int j, T alpha, Double epsilon) {
    // When alpha is very small, using the antiderivative can lead to bad
    // cancellation and loss of precision, so we use the taylor series
    // expansion for exp(2 pi alpha t) instead, and integrate term by term
    // using antiderivatives.

    // The taylor expansion gives
    //
    // H(j, alpha) = sum_{m=0}^\infty (-2 pi alpha)^m / (m! (j + m + 1) )
    //
    //int N = max(ceil(-log(epsilon)), ceil(2 * PI * alpha_0 * E * E));; // the number of terms we will use in the taylor expansion

    T S = 0.0;
    
    Double error = epsilon + 1.0;
    int m = 0;
    T alpha_power = 1.0;
    while(error > epsilon/2.0) {
        if(m > 0) {
            alpha_power *= -alpha;
        }
        T z = two_pi_over_factorial_power(m) * alpha_power / Double(j + m + 1);
        S = S + z;
        error = abs(z);
        m++;
    }

    return S;
}

template<typename T> T H_method4(int j, T alpha, Double epsilon) {
    // Compute H(j, alpha) using a continued fraction expansion
    //
    // This code is largely copied from lcalc.
    T P1 = 1.0;
    T P2 = (T)(j + 1.0);
    T P3 = 0.0;
    
    T Q1 = 0.0;
    T Q2 = 1.0;
    T Q3 = 0.0;
    
    T u = PI * alpha;
    Double z = j + 1;
    T w = 2.0 * PI * alpha;

    int n=0;
    Double error = epsilon + 1;
    while( (error > epsilon || n < 3) && n < 1000000) {
        n++;
        P3=(z+n)*P2-(z+(n-1)*.5)*w*P1;
        Q3=(z+n)*Q2-(z+(n-1)*.5)*w*Q1;

        P1=P2;P2=P3;
        Q1=Q2;Q2=Q3;

        n++;
        P3=(z+n)*P2+(Double)n*u*P1;
        Q3=(z+n)*Q2+(Double)n*u*Q1;

        P1=P2;P2=P3;
        Q1=Q2;Q2=Q3;

        //to prevent overlow
        if(n%8==0&&(real(P2)>1.e50||real(P2)<-1.e50||imag(P2)>1.e50||imag(P2)<-1.e50)){
            P1=P1*(Double)(1.e-50);
            P2=P2*(Double)(1.e-50);
            Q1=Q1*(Double)(1.e-50);
            Q2=Q2*(Double)(1.e-50);

        }

        error = abs( ((P1 * Q2) - (P2 * Q1))/(Q1 * Q2) );

    }

    T g=P2/Q2;

    if(n>999999){
         cout << "Mofu. Continued fraction for g(z,w) failed to converge. z = "
         << z << "  w = " << w << endl;
         //exit(1);
    }

    g = EXP(-w)/g;
    return g;
}

template<typename T> Complex H(int j, T alpha, Double epsilon) {
    //
    // Compute the integral int_0^1 t^j exp(-2 pi alpha t) dt
    // We have three different methods to compute this depending
    // on the range of input.
    //

    if((j + 1) * epsilon > 1.0) {
        return 0.0;
    }

    const Double alpha_0 = 1/(2*PI);

    Double norm_alpha = norm(alpha);

    //if(abs(alpha) < alpha_0) {
    if(norm_alpha < alpha_0 * alpha_0) {
            return H_method2(j, alpha, epsilon);
    }
    //else if(abs(2 * PI * alpha) > j/2) {
    else if(4 * PI * PI * norm_alpha > j * j / 4) {
            return H_method1(j,  alpha);
    }
    else {
            return H_method4(j, alpha, epsilon);
    }
   
}

//
// The following are old methods that are no longer used, but kept around
// in case we want to revisit them.
//

Complex H_method3(int j, Complex alpha, Double epsilon) {
    //
    // THIS IS CURRENTLY UNUSED.
    //
    // We use a change of variables to rewrite the integral, and compute it as
    //
    // (1/M^{j+1}) sum_{m=0}^{M-1} exp( 2 pi i alpha m / M ) sum_{r=0}^j (j choose r) m^r H(j-r, alpha/M)
    //
    // Here, in the inner sum, alpha/M will always be small enough that H(j-r, alpha/M) is computed
    // with method 2.
    //

    int M = to_int(ceil(abs(2 * PI * alpha)));

    Complex S = (Complex)0;

    Complex H_table[max_j + 1];
    for(int m = 0; m <= j; m++) {
        H_table[m] = H_method2(m, alpha/(Double)M, j * epsilon);
    }


    Complex exponential_multiplication_factor = EXP(-2.0 * PI * alpha/ (Double)M);
    Complex exponential_factor = (Complex)1.0/exponential_multiplication_factor;

    for(int m = 0; m <= M - 1; m++) {
        Complex S2 = (Complex)0;
        Double m_power = 1;
        for(int r = 0; r <= j; r++) {
            if(r > 0) {
                m_power *= m;
            }
            Complex z = binomial_coefficient(j, r) * m_power * H_table[j-r];
            S2 = S2 + z;
        }

        exponential_factor *= exponential_multiplication_factor;
        Complex z = exponential_factor * S2; // Here we should be multiplying S2 by exp(-2 pi alpha m/M), if we are doing things correctly.
        S = S + z;


    }
    S = S/(Double)pow((Double)M,j + 1);
    return S;
}
Complex H_method5(int j, Complex alpha, Double epsilon) {
    //
    // THIS IS CURRENTLY UNUSED.
    //

//    Complex alpha_0 = Complex( n/(Double)N, m/(Double)M );
    Complex alpha_0 = alpha + 1.0;
    Complex difference = alpha - alpha_0;
    Complex difference_power = 1;

//    Double error = epsilon + 2;

    Double error = 12;

    Complex Z = 0;
    int r = 0;
    Double one_over_j_plus_1 = 1.0/(j + 1.0);
    while(error > 10) {
        Complex z1 = two_pi_over_factorial_power(r) * difference_power;
        Z = Z + (Double)minus_one_power(r) * z1; // * H_method4(j + r, alpha_0, epsilon/20);
        error = abs(z1) * one_over_j_plus_1;
        difference_power *= difference;
        r++;
    }

    return Z;
}
