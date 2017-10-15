#ifndef QUAD_MPFR
#define QUAD_MPFR


#include "quad-imp.h"
#include <mpfr.h>

#if GMP_LIMB_BITS != 64
# error "Too bad"
#endif



inline __attribute__((always_inline)) static void mpfr_set_float128 (mpfr_t a, const __float128 val)
{
  ieee854_float128 f;
  f.value = val;

  int exp = f.ieee.exponent;
  int subnormal = 0;


  uint64_t h = f.ieee.mant_high;
  uint64_t l = f.ieee.mant_low;

  MPFR_SIGN(a) = f.ieee.negative ? -1 : +1;

  // Special case for zero exponent
  if (exp == 0)
  {
    if (h == 0 && l == 0)
    {
      a->_mpfr_exp = __MPFR_EXP_ZERO;
      return;
    }
    else
    {
      subnormal = 1;
      exp = 2 - IEEE854_FLOAT128_BIAS;
    }
  }
  else if (exp == 0x7fff)
  {
    if (h == 0 && l == 0)
    {
      a->_mpfr_exp = __MPFR_EXP_INF;
      return;
    }
    else
    {
      a->_mpfr_exp = __MPFR_EXP_NAN;
      return;
    }
  }
  else
    exp -= (IEEE854_FLOAT128_BIAS - 1);

  if (subnormal)
  {
    while (!(h & (UINT64_C(1) << (63 - 15))))
    {
      exp--;
      h <<= 1;
      h |= (l & (UINT64_C(1) << 63)) >> 63;
      l <<= 1;
    }
  }

  a->_mpfr_exp = exp;
  a->_mpfr_d[1] = INT64_C(1) << 63 | h << 15 | l >> 49;
  a->_mpfr_d[0] = l << 15;
}

/*
inline static qd_real mpfr_get_qd2(const mpfr_t a){
  qd_real temp; 
  if(mpfr_zero(a))
    return (qd_real) 0; 
  
  temp.x[0] = a->_mpfr_d[0];
  temp.x[1] = a->_mpfr_d[1];
  temp.x[2] = a->_mpfr_d[2];
  temp.x[3] = a->_mpfr_d[3];
  temp *= pow((qd_real) 10, a->_mpfr_exp); 
  return temp; 
}
*/

inline __attribute__((always_inline)) static __float128 mpfr_get_float128 (const mpfr_t a)
{
  ieee854_float128 f;

  if (mpfr_nan_p(a))
  {
    f.ieee.exponent = 0x7fff;
    f.ieee.mant_high = f.ieee.mant_low = 1;
    return f.value;
  }

  if (mpfr_inf_p(a))
  {
    f.ieee.exponent = 0x7fff;
    f.ieee.mant_high = f.ieee.mant_low = 0;
    f.ieee.negative = (MPFR_SIGN(a) == -1) ? 1 : 0;
    return f.value;
  }
  
  if (mpfr_zero_p(a))
  {
    f.ieee.exponent = 0;
    f.ieee.mant_high = f.ieee.mant_low = 0;
    f.ieee.negative = (MPFR_SIGN(a) == -1) ? 1 : 0;
    return f.value;
  }

  f.ieee.negative = (MPFR_SIGN(a) == -1) ? 1 : 0;

  uint64_t x = a->_mpfr_d[1];
  uint64_t y = a->_mpfr_d[0];

  if (a->_mpfr_exp <= -IEEE854_FLOAT128_BIAS + 1)
  {
    // Subnormal
    int i;
    for (i = 0; i <= - a->_mpfr_exp - IEEE854_FLOAT128_BIAS + 1; i++)
    {
      y >>= 1;
      y |= (x & 0x1) << 63;
      x >>= 1;
    }

    f.ieee.exponent = 0;
  }
  else
    f.ieee.exponent = a->_mpfr_exp + (IEEE854_FLOAT128_BIAS - 1);

  f.ieee.mant_high = x >> 15;  // Or: f.ieee.mant_high = (x << 1) >> 16;
  f.ieee.mant_low = (x << 49) | (y >> 15);

  return f.value;
}

/*


inline static __float128
nextafterq (__float128 x, __float128 y)
{
  MPFR_DECL_INIT (a, 113);
  MPFR_DECL_INIT (b, 113);
  mpfr_set_float128 (a, x);
  mpfr_set_float128 (b, y);
  mpfr_nexttoward (a, b);
  return mpfr_get_float128 (a);
}
*/

#endif
