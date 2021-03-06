#ifndef QUAD_IMP_H
#define QUAD_IMP_H

#include <stdint.h>
#include <stdlib.h>


// Prototypes for internal functions
//extern int32_t rem_pio2q (__float128, __float128 *);
//extern void __kernel_sincosq (__float128, __float128, __float128 *, __float128 *, int);
//extern __float128 __kernel_sinq (__float128, __float128, int);
//extern __float128 __kernel_cosq (__float128, __float128);



// Frankly, if you have __float128, you have 64-bit integers, right?
#ifndef UINT64_C
# error "No way!"
#endif


// If we don't have macros to know endianess, assume little endian
#if !defined(__BIG_ENDIAN__) && !defined(__LITTLE_ENDIAN__)
# define __LITTLE_ENDIAN__ 1
#endif


// Main union type we use to manipulate the floating-point type
typedef union
{
  __float128 value;

  struct
  {
#if __BIG_ENDIAN__
    unsigned negative:1;
    unsigned exponent:15;
    uint64_t mant_high:48;
    uint64_t mant_low:64;
#endif
#if __LITTLE_ENDIAN__
    uint64_t mant_low:64;
    uint64_t mant_high:48;
    unsigned exponent:15;
    unsigned negative:1;
#endif
  } ieee;

  struct
  {
#if __BIG_ENDIAN__
    uint64_t high;
    uint64_t low;
#endif
#if __LITTLE_ENDIAN__
    uint64_t low;
    uint64_t high;
#endif
  } words64;

  struct
  {
#if __BIG_ENDIAN__
    uint32_t w0;
    uint32_t w1;
    uint32_t w2;
    uint32_t w3;
#endif
#if __LITTLE_ENDIAN__
    uint32_t w3;
    uint32_t w2;
    uint32_t w1;
    uint32_t w0;
#endif
  } words32;

  struct
  {
#if __BIG_ENDIAN__
    unsigned negative:1;
    unsigned exponent:15;
    unsigned quiet_nan:1;
    uint64_t mant_high:47;
    uint64_t mant_low:64;
#endif
#if __LITTLE_ENDIAN__
    uint64_t mant_low:64;
    uint64_t mant_high:47;
    unsigned quiet_nan:1;
    unsigned exponent:15;
    unsigned negative:1;
#endif
  } nan;

} ieee854_float128;


/* Get two 64 bit ints from a long double.  */
#define GET_FLT128_WORDS64(ix0,ix1,d)  \
do {                                   \
  ieee854_float128 u;                  \
  u.value = (d);                       \
  (ix0) = u.words64.high;              \
  (ix1) = u.words64.low;               \
} while (0)

/* Set a long double from two 64 bit ints.  */
#define SET_FLT128_WORDS64(d,ix0,ix1)  \
do {                                   \
  ieee854_float128 u;                  \
  u.words64.high = (ix0);              \
  u.words64.low = (ix1);               \
  (d) = u.value;                       \
} while (0)

/* Get the more significant 64 bits of a long double mantissa.  */
#define GET_FLT128_MSW64(v,d)          \
do {                                   \
  ieee854_float128 u;                  \
  u.value = (d);                       \
  (v) = u.words64.high;                \
} while (0)

/* Set the more significant 64 bits of a long double mantissa from an int.  */
#define SET_FLT128_MSW64(d,v)          \
do {                                   \
  ieee854_float128 u;                  \
  u.value = (d);                       \
  u.words64.high = (v);                \
  (d) = u.value;                       \
} while (0)

/* Get the least significant 64 bits of a long double mantissa.  */
#define GET_FLT128_LSW64(v,d)          \
do {                                   \
  ieee854_float128 u;                  \
  u.value = (d);                       \
  (v) = u.words64.low;                 \
} while (0)


#define IEEE854_FLOAT128_BIAS 0x3fff


#endif

