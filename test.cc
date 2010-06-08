#include "theta_sums.h"
#include "log.h"
#include "zeta.h"

#include <ctime>
#include <iostream>
#include <iomanip>

using namespace std;

inline double random_double() {
    return (double)rand()/(double)RAND_MAX;   
}
inline complex<double> random_complex() {
    return complex<double>(random_double(), random_double());
}

int test_fastlog() {
    const int number_of_tests = 10000;

    cout << "Testing fastlog() on " << number_of_tests << " uniformly random numbers between 1 and 1000000...";

    int failures1 = 0;
    int smaller1 = 0;
    int exact1 = 0;

    for(int n = 0; n < number_of_tests && failures1 <= 10; n++) {
        double d = (double)rand()/(double)RAND_MAX * 999999.0 + 1.0;
        int a = fastlog(d);
        int b = (int)floor(log(d));
        if(a != b && a+1 != b) {
            failures1++;
            if(failures1 < 11) {
                cout << endl;
                cout << "fastlog(" << d << ") gives wrong result. Got: " << a << ". Expected: " << b << ".";
            }
        }
        if(a == b)
            exact1++;
        else
            smaller1++;
    }
    if(failures1) {
        cout << "done." << endl;
        cout << "**************************************************" << endl;
        cout << "TEST FAILURES FOUND IN fastlog()." << endl;
        cout << "**************************************************" << endl;
    }
    else {
        cout << "OK." << endl;
        cout << "       Answer was exact " << exact1 << " times." << endl;
        cout << "  Answer was off by one " << smaller1 << " times." << endl;
    }

    cout << "Testing fastlog() on " << number_of_tests << " uniformly random numbers between 0 and 1...";

    int failures2 = 0;
    int smaller2 = 0;
    int exact2 = 0;

    for(int n = 0; n < number_of_tests && failures2 <= 10; n++) {
        double d = (rand() + 1.0)/(RAND_MAX + 1.0);
        int a = fastlog(d);
        int b = (int)floor(log(d));
        if(a != b && a + 1 != b) {
            failures2++;
            if(failures2 < 11) {
                cout << endl;
                cout << "fastlog(" << d << ") gives wrong result. Got: " << a << ". Expected: " << b << ".";
            }
        }
        if(a == b)
            exact2++;
        else
            smaller2++;
    }
    if(failures2) {
        cout << "done." << endl;
        cout << "**************************************************" << endl;
        cout << "TEST FAILURES FOUND IN fastlog()." << endl;
        cout << "**************************************************" << endl;
    }
    else {
        cout << "OK." << endl;
        cout << "       Answer was exact " << exact2 << " times." << endl;
        cout << "  Answer was off by one " << smaller2 << " times." << endl;
    }

    cout << "Testing fastlog() on " << number_of_tests << " uniformly random numbers between 0 and .0000001...";

    int failures3 = 0;
    int smaller3 = 0;
    int exact3 = 0;

    for(int n = 0; n < number_of_tests && failures3 <= 10; n++) {
        double d = (rand() + 1.0)/(RAND_MAX + 1.0) * .0000001;
        int a = fastlog(d);
        int b = (int)floor(log(d));
        if(a != b && a + 1 != b) {
            failures3++;
            if(failures3 < 11) {
                cout << endl;
                cout << "fastlog(" << d << ") gives wrong result. Got: " << a << ". Expected: " << b << ".";
            }
        }
        if(a == b)
            exact3++;
        else
            smaller3++;
    }
    if(failures3) {
        cout << "done." << endl;
        cout << "**************************************************" << endl;
        cout << "TEST FAILURES FOUND IN fastlog()." << endl;
        cout << "**************************************************" << endl;
    }
    else {
        cout << "OK." << endl;
        cout << "       Answer was exact " << exact3 << " times." << endl;
        cout << "  Answer was off by one " << smaller3 << " times." << endl;
    }
    

    return failures1 + failures2 + failures3;
}




int test_fastlog2() {
    const int number_of_tests = 10000;

    cout << "Testing fastlog2() on " << number_of_tests << " uniformly random numbers between 1 and 1000000...";

    int failures1 = 0;

    for(int n = 0; n < number_of_tests && failures1 <= 10; n++) {
        double d = (double)rand()/(double)RAND_MAX * 999999.0 + 1.0;
        int a = fastlog2(d);
        int b = (int)floor(log2(d));
        if(a != b) {
            failures1++;
            if(failures1 < 11) {
                cout << endl;
                cout << "fastlog2(" << d << ") gives wrong result. Got: " << a << ". Expected: " << b << ".";
            }
        }
    }
    if(failures1) {
        cout << "done." << endl;
        cout << "**************************************************" << endl;
        cout << "TEST FAILURES FOUND IN fastlog2()." << endl;
        cout << "**************************************************" << endl;
    }
    else {
        cout << "OK." << endl;
    }

    cout << "Testing fastlog2() on " << number_of_tests << " uniformly random numbers between 0 and 1...";

    int failures2 = 0;

    for(int n = 0; n < number_of_tests && failures2 <= 10; n++) {
        double d = (rand() + 1.0)/(RAND_MAX + 1.0);
        int a = fastlog2(d);
        int b = (int)floor(log2(d));
        if(a != b) {
            failures2++;
            if(failures2 < 11) {
                cout << endl;
                cout << "fastlog2(" << d << ") gives wrong result. Got: " << a << ". Expected: " << b << ".";
            }
        }
    }
    if(failures2) {
        cout << "done." << endl;
        cout << "**************************************************" << endl;
        cout << "TEST FAILURES FOUND IN fastlog2()." << endl;
        cout << "**************************************************" << endl;
    }
    else {
        cout << "OK." << endl;
    }

    cout << "Testing fastlog2() on " << number_of_tests << " uniformly random numbers between 0 and .00001...";

    int failures3 = 0;

    for(int n = 0; n < number_of_tests && failures3 <= 10; n++) {
        double d = (rand() + 1.0)/(RAND_MAX + 1.0) * .00001;
        int a = fastlog2(d);
        int b = (int)floor(log2(d));
        if(a != b) {
            failures3++;
            if(failures3 < 11) {
                cout << endl;
                cout << "fastlog2(" << d << ") gives wrong result. Got: " << a << ". Expected: " << b << ".";
            }
        }
    }
    if(failures3) {
        cout << "done." << endl;
        cout << "**************************************************" << endl;
        cout << "TEST FAILURES FOUND IN fastlog2()." << endl;
        cout << "**************************************************" << endl;
    }
    else {
        cout << "OK." << endl;
    }


    return failures1 + failures2 + failures3;
}

int test_theta_algorithm(int number_of_tests) {
    const int j_max = 18;

    Complex v[j_max + 1];

    cout << "Testing theta algorithm with various random parameters " << number_of_tests << " times." << endl;
    Double maxerror = 0.0;
    for(int n = 0; n < number_of_tests; n++) {
        Double a = (double)rand()/(double)RAND_MAX * 20.0 - 10.0;
        Double b = (double)rand()/(double)RAND_MAX * 20.0 - 10.0;

        int K = (int)((double)rand()/(double)RAND_MAX * 500.0 + 10000);
        int j = (int)((double)rand()/(double)RAND_MAX * j_max);
    
        for(int k = 0; k <= j; k++) {
            v[k] = random_complex() * 2.0 - complex<double>(1.0, 1.0);
        }

        Complex S1 = compute_exponential_sums(a, b, j, K, v, pow(2.0, -29));
        Complex S2 = compute_exponential_sums(a, b, j, K, v, pow(2.0, -29), 1);

        Double error = abs(S1 - S2);
        maxerror = max(error, maxerror);

        cout << "Test " << n << ": a = " << a << ", b = " << b << ", j = " << j << ", K = " << K << ": log2(error) = " << log2(error) << endl;
    }
    cout << "Largest error was " << maxerror << "; log2(maxerror) = " << log2(maxerror) << endl;

    return 0;
}

double time_theta_algorithm(int j, int K) {
    Complex v[j + 1];
    for(int k = 0; k <= j; k++) {
        v[j] = 1.0;
    }

    clock_t start_time = clock();

    Double epsilon = exp(-20);

    int n = 0;
    Complex z1 = 0.0;

    const int number_of_tests = 10000;

    cout << "Timing theta_algorithm with K = " << K << " and j = " << j << endl;
    cout << "   Running approximately 10000 iterations total." << endl;

    for(Double a = 0; a < .5; a += .5/(number_of_tests/1000.0) ) {
        for(Double b = 1.0/((Double)2 * K * K); b <= 1.0; b += 1.0/1000.0) {
            n++;
            if(n % 500 == 0) {
                cout << "   Running iteration number " << n << " with a = " << a << " b = " << b << endl;
            }
            z1 += compute_exponential_sums(a, b, j, K, v, epsilon);
        }
    }
    cout << "Sum was " << z1 << endl;

    clock_t end_time = clock();

    double elapsed_time = (double)(end_time - start_time)/(double)CLOCKS_PER_SEC;
    cout << "Number of seconds for this run was " << elapsed_time << endl;

    return elapsed_time;
    
}


double time_zeta_block(int K, int samples) {
    
    mpfr_t t;
    mpz_t v;
    mpz_t inc;
    mpz_init(v);
    mpz_init(inc);
    mpfr_init2(t, 150);
    mpz_set_str(v, "400000000000000", 10);
    mpfr_set_str(t, "1e30", 10, GMP_RNDN);

    mpz_div_ui(inc, v, samples);

    Complex S = 0;
    Complex Z[30];
    
    compute_taylor_coefficients(t, Z);
    
    cout << "Timing zeta_block() with K = " << K << " and " << samples << " samples. ...";
    
    cout.flush();

    clock_t start_time = clock();

    for(int k = 0; k < samples; k++) {
        mpz_add(v, v, inc);
        S += zeta_block(v, K, t, Z);
    }
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time)/(double)CLOCKS_PER_SEC;
    cout << elapsed_time << " seconds." << endl;
    
    cout << S << endl;

    return elapsed_time;
}

int test_exp_itlogn(gmp_randstate_t state) {
    mpfr_t t;
    mpfr_init2(t, 158);
    mpfr_set_str(t, "1e30", 10, GMP_RNDN);

    Complex z1, z2;
    
    mpz_t n, m;
    mpz_init(n);
    mpz_init(m);
    mpz_set_str(m, "100000000000000", 10);

    mpfr_t twopi;
    mpfr_init2(twopi, mpfr_get_prec(t));
    mpfr_const_pi(twopi, GMP_RNDN);
    mpfr_mul_ui(twopi, twopi, 2, GMP_RNDN);

    mpfr_t w1;
    mpfr_init2(w1, mpfr_get_prec(t));
    
    create_exp_itlogn_table(t);

    mpz_set_str(n, "100000000", 10);

//    cout << exp_itlogn(n) << endl;
    
    z1 = 0;
    z2 = 0;

    int failures = 0;

    cout << "Testing exp_itlogn() against mpfr 200000 times ...";
    cout.flush();

    for(int k = 1; k <= 200000; k++) {
        mpz_urandomm(n, state, m);
        mpfr_set_z(w1, n, GMP_RNDN);

        mpfr_log(w1, w1, GMP_RNDN);
        mpfr_mul(w1, w1, t, GMP_RNDN);
        mpfr_fmod(w1, w1, twopi, GMP_RNDN);
        z1 = exp(I * mpfr_get_d(w1, GMP_RNDN));

        z2 = exp_itlogn4(n);

        Double error = abs(z1 - z2);

        if(error > 1e-14) {
            cout << endl;
            cout << "Large error found with n = " << n << ". Error was " << abs(z1 - z2);
            failures++;
        }
        if(failures == 11) {
            break;
        }
    }
    if(failures) {
        cout << "done." << endl;
    }
    else {
        cout << "OK." << endl;
    }

    return failures;
}

int time_exp_itlogn() {
    mpfr_t t;
    mpfr_init2(t, 158);
    mpfr_set_str(t, "1e30", 10, GMP_RNDN);

    Complex z2;
    
    mpz_t n;
    mpz_init(n);

    create_exp_itlogn_table(t);

    mpz_set_str(n, "100000000000", 10);

//    cout << exp_itlogn(n) << endl;
    
    z2 = 0;

    const int number_of_iterations = 2000000;
    cout << "Timing exp_itlogn() over " << number_of_iterations << " iterations ...";
    cout.flush();
    
    clock_t start_time = clock();

    for(int k = 0; k < number_of_iterations; k++) {
        mpz_add_ui(n, n, 1u);
        z2 += exp_itlogn4(n);
    }
    cout << z2 << "...";

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time)/(double)CLOCKS_PER_SEC;
    cout << elapsed_time << " seconds." << endl;

    return elapsed_time;
}

int time_exp_itlogn_mpfr() {
    mpfr_t t;
    mpfr_init2(t, 158);
    mpfr_set_str(t, "1e30", 10, GMP_RNDN);

    Complex z1;
    
    mpz_t n;
    mpz_init(n);

    mpfr_t twopi;
    mpfr_init2(twopi, mpfr_get_prec(t));
    mpfr_const_pi(twopi, GMP_RNDN);
    mpfr_mul_ui(twopi, twopi, 2, GMP_RNDN);

    mpfr_t w1;
    mpfr_init2(w1, mpfr_get_prec(t));
    
    create_exp_itlogn_table(t);

    mpz_set_str(n, "100000000000", 10);

//    cout << exp_itlogn(n) << endl;
    
    z1 = 0;

    const int number_of_iterations = 2000000;
    cout << "Timing exp_itlogn using mpfr over " << number_of_iterations << " iterations ...";
    cout.flush();

    clock_t start_time = clock();
    for(int k = 0; k <= number_of_iterations; k++) {
        mpz_add_ui(n, n, 1u);
        mpfr_set_z(w1, n, GMP_RNDN);

        mpfr_log(w1, w1, GMP_RNDN);
        mpfr_mul(w1, w1, t, GMP_RNDN);
        mpfr_fmod(w1, w1, twopi, GMP_RNDN);
        z1 += exp(I * mpfr_get_d(w1, GMP_RNDN));
    }

    cout << z1 << "...";

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time)/(double)CLOCKS_PER_SEC;
    cout << elapsed_time << " seconds." << endl;

    return elapsed_time;

}





int main() {
    unsigned int seed = time(NULL);
    cout << "Seeding rand() and gmp with " << seed << "." << endl;
    srand(seed);
    
    gmp_randstate_t rand_state;
    gmp_randinit_default(rand_state);
    gmp_randseed_ui(rand_state, seed);

    cout << setprecision(15);
    //test_fastlog2();
    //test_fastlog();
    //test_theta_algorithm(10);
    //time_theta_algorithm(18, 10010);
    //time_zeta_block(36000, 5000);
    test_exp_itlogn(rand_state);
    time_exp_itlogn();
    time_exp_itlogn_mpfr();

    return 0;
}