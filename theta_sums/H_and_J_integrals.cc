#include "theta_sums.h"
#include "precomputed_tables.h"
#include "log.h"

#include <iostream>
#include <string>
#include <fstream>
#include <cmath>

#include <sys/mman.h>
#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace std;

Complex H_Integral_0(int j, Double a, int M, Double epsilon) {
    //
    // Compute the integral int_0^1 t^j exp(-2 pi a t) (1 - exp(-2 pi M t))/(exp(2 pi t) - 1)dt,
    // which can be written (by expanding the geometric sum) as
    //
    // sum_{m=1}^M int_0^1 t^j exp(-2 pi (a + m) t) dt
    //
    //
    // In terms of our function H(j, a, epsilon), this is
    //
    // sum_{m=1}^M H(j, a + m, epsilon/(2 j + 1)
    //
    // We don't want to compute all of these terms directly, however, because
    // M can be large. We separate this into the case of small m, where
    // we compute directly, and then large m, where the antiderivative
    // can be approximated simply. We compute "simple" approximation
    // with Euler-Maclaurin summation.
    //
    // small m is just m < C = max(j, ceil(-log(epsilon)/2pi)
    //

    Complex S = (Complex)0;

    //int C = max(to_int(ceil(j/(2 * PI * E))), to_int(ceil(-LOG(epsilon)/(2 * PI))) );
    int C = max(to_int(ceil(j/(2 * PI * E))), to_int(ceil(-fastlog(epsilon)/(2 * PI))) );
    //int C = ceil(max(j/E, -(Double)(fastlog(epsilon)))/2 * PI) ;

    if(M != -1) {
        C = min(M, C);
    }

    for(int m = 1; m <= C; m++) {
        Complex z = H(j, a + m, epsilon/(C + 1));
        S = S + z;
    }

    if(C == M) {
        return S;
    }
    
    Double z = factorial(j)/two_pi_power(j + 1);

    S = S + z * sum_of_offset_inverse_powers(a, C + 1, M, j+1, epsilon/z);

    return S;

}

Complex J_Integral_0(Double a, Double b, int j, int M, int K, theta_cache * cache, Double epsilon) {
    //
    // Compute the integral int_0^1 exp(-2 pi a t - 2 pi i b t^2) (1 - exp(-2 pi M t))/(exp(2 pi t) - 1) dt
    //
    // We can do this by expanding the denominator in a geometric series
    // and doing a taylor expansion in b, which puts this integral in the form
    //
    // sum_{r=0}^\infty ((2 pi i b)^r)/r! \sum_{m=1}^M int_0^1 t^j exp(-2 pi(m + a)t)dt
    //
    // -1 corresponds to infinity

    Double K_pow_j;
    Double K_pow_minus_j;
    if(cache) {
        K_pow_j = K_power(j, cache);
        K_pow_minus_j = K_power(-j, cache);
    }
    else {
        K_pow_j = pow(K, j);
        K_pow_minus_j = 1.0/K_pow_j;
    }

    

    //if(2 * pow(K, -j) * log(abs((Double)M) + 10) < epsilon) {
    if( 3 + fastlog2(K_pow_minus_j) + fastlog2(fastlog(abs(M) + 10.0)) < fastlog2(epsilon)) {
        return 0.0;
    }
    epsilon = epsilon * K_pow_j;

    Complex S = (Complex)0;

    int r = 0;
    Double error = 2 * epsilon;
    
    if(b == 0) {
        Complex answer = H_Integral_0(j, a, M, epsilon)*K_pow_minus_j;
        return answer;
    }
    
    int N = max(1, to_int(ceil(-fastlog(epsilon))));    // an approximation of the number of terms
                                                    // we compute in the taylor expansion

    Double b_power = (Double)1.0;
    while(error > epsilon) {
        Complex z = minus_I_power(r) * two_pi_over_factorial_power(r) * b_power;
        error = abs(z);
        z *= H_Integral_0(2 * r + j, a, M, epsilon/(error *N));  // TODO: We really don't need to compute this term to
                                                                                // this much precision usually. We really should figure
                                                                                // our how many terms we are going to compute
                                                                                // and then divide appropriately.
                                                                                //
        S = S + z;
        b_power *= b;
        r++;
    }

    S = S * K_pow_minus_j;

    return S;
}


Complex J_Integral_1(Double a, Double b, int j, int M, int K, theta_cache * cache, Double epsilon) {
    //
    // Compute the integral int_1^K exp(-2 pi a t - 2 pi i b t^2)(1 - exp(-2 pi M t))/(exp(2 pi t) - 1) dt
    //
    // If M == -1, then we treat it as positive infinity.
    //
    
    // We truncate the integral at L, where L is given by

    //int L = min(K, max(1, to_int(ceil(-LOG(epsilon)/(2 * PI * (1 + a))))));

    int L = ceil(-fastlog(epsilon) * (1/(2 * PI)) * inverse((int)(1 + a)));
    if(L <= 1) {
        return 0.0;
    }
    L = max(1.0, L - j * fastlog( (Double)K * inverse(L + 1)) * (1.0/(2 * PI)) );
    L = min(K, L);


    if(L <= 1) {
        return 0.0;
    }

    // Now we compute the integral as a sum over unit length integrals.
    // On each unit length interval, we do a change of variables to get
    // the range of integration between 0 and 1, and reduce the computation
    // to a sum of exponentials times G()

    Complex S = (Complex)0;

    //Double one_over_K_to_the_j = pow( (Double)K, -j);
    Double one_over_K_to_the_j;
    if(cache)
        one_over_K_to_the_j = K_power(-j, cache);
    else
        one_over_K_to_the_j = pow(K, -j);

    //if(2 * pow(L, j) * one_over_K_to_the_j * log(abs((Double)M) + 10) < epsilon) {
    if( 5 + j * fastlog2(L) + fastlog2(one_over_K_to_the_j) - 2 * PI * (a + 1) * log2(E) < fastlog2(epsilon)) {
        return 0.0;
    }

    Double exp_minus_twopi = EXP(-2.0 * PI);
    Double exp_minus_twopi_n = 1.0;

    //Double a_factor = exp(-2.0 * PI * (1 + a));
    //Double a_multiplier = a_factor;

    for(Double n = (Double)1; n <= L - 1; n = n + 1) {
        exp_minus_twopi_n *= exp_minus_twopi;
        int end_point;
        if(M == -1) {
            end_point = to_int(ceil(-fastlog(epsilon/(2 * L))/(2 * PI * n) -  j * fastlog((Double)K/(n + 1))/(2 * PI) ));
        }
        else {
            end_point = min(M, to_int(ceil(-fastlog(epsilon/(2 * L))/(2 * PI * n) - j * fastlog((Double)K/(n + 1))/(2 * PI))  )  );
            //end_point = min(M, to_int(ceil(-log(epsilon/(2 * L))/(2 * PI * n)    )    )      );
        }

        end_point = max(1, end_point);

        //end_point = max(1, (int)ceil(end_point - j * log((Double)K/(n + 1))/(2 * PI)));
        //end_point = max(1, (int)(end_point - j * fastlog((Double)K/(n + 1))/(2 * PI)));

        Complex S1 = 0;

        Complex x = EXP(-2.0 * PI * n * (1.0 + a + I * b * n));
        //Double d = -2 * PI * b * n * n;
        //Complex x = a_factor * Complex(cos(d), sin(d));

        for(Double m = (Double)1; m <= end_point; m = m + 1) {
            
            if(m > 1)
                x = x * exp_minus_twopi_n;
            //Complex z =  G(I*(m + a + (Complex)2.0 * I * b * n), -b, n, j, epsilon/(abs(x) * end_point * Double(L - 1) * one_over_K_to_the_j));
            Complex z =  G(I*(m + a + (Complex)2.0 * I * b * n), -b, n, j, epsilon/(abs(x) * end_point * Double(L - 1) * one_over_K_to_the_j));
            z *= x;
            S1 = S1 + z;
        }
        S1 = S1 * one_over_K_to_the_j;
        S = S + S1;
        //a_factor *= a_multiplier;
    }


    return S;

}

Complex H_Integral_2(int j, Double a1, Double a2, Double epsilon) {

    Complex S = (Complex)0;

    int C = max(to_int(ceil(j/(2*PI*E))), to_int(ceil(-fastlog(epsilon)/(2 * PI) ) ));

    for(int m = 1; m <= C; m++) {
        Complex z = H(j, m + a1, epsilon/(C + 1)) - H(j, m + a2, epsilon/(C + 1));
        S = S + z;
    }

    S = S + factorial(j)/two_pi_power(j + 1) * infinite_sum_of_differenced_inverse_powers(a1, a2, C + 1, j+1, epsilon);

    return S;

}

Complex J_Integral_2(Double a1, Double a2, Double b, theta_cache * cache, Double epsilon) {
    //
    // Compute the integral int_0^1 exp(-2 pi i b t^2) (exp(-2 pi a1 t) - exp(-2 pi a2 t))(exp(2 pi t) - 1) dt,
    //
    // which is equal to 
    //
    // lim_{M --> oo} J_Integral_0(a1, b, M) - J_Integral_0(a2, b, M)
    //

    if(a2 > a1) {
        //if (2.0 * ( (a2 - a1)/a2 * log(1 + a2) ) < epsilon) {
        if(1 + fastlog2( (a2 - a1)/a2 * (fastlog(1 + a2) + 1.0)) < fastlog2(epsilon)) {
            return 0.0;
        }
    }
    else {
        //if (2.0 * ( (a1 - a2)/a1 * log(1 + a1) ) < epsilon) {
        if(1 + fastlog2( (a1 - a2)/a1 * (fastlog(1 + a1) + 1.0)) < fastlog2(epsilon)) {
            return 0.0;
        }
    }

    Complex S = (Complex)0;

    int r = 0;
    Double error = 2 * epsilon;
    int sign = -1;
    
    if(b == 0) {
        Complex answer = H_Integral_2(0, a1, a2, epsilon);
        return answer;
    }
   
    int N = max(1, to_int(-fastlog(epsilon)));

    Complex Ib_power = (Complex)1.0/(I * b);
    while(error > epsilon) {
        sign = -sign;
        Ib_power *= (I * b);
        Complex z = sign * two_pi_over_factorial_power(r) * Ib_power;
        error = abs(z);
        
        z *= H_Integral_2(2 * r, a1, a2, epsilon/(error * N));  // TODO: We really don't need to compute this term to
                                                                                // this much precision usually. We really should figure
                                                                                // our how many terms we are going to compute
                                                                                // and then divide appropriately.
        S = S + z;
        r++;
    }



    return S;

}
