#                                               -*- utoconf -*-
# Process this file with autoconf to produce a configure script.

# Should we abort if we are  a 64bit architecture?

AC_PREREQ([2.69])

AC_INIT([zetacalc], [0.1])
AC_CONFIG_SRCDIR([zetacalc.cc])
AC_CONFIG_HEADERS([include/config.h])

# Checks for programs.
export HAVE_MPI=1
export HAVE_CUDA=1
AC_DEFINE([HAVE_CUDA], [1], [Do we have CUDA?])
AC_DEFINE([HAVE_MPI], [1], [Do we have MPI?])

AC_CHECK_SIZEOF(unsigned long) 
AC_CHECK_SIZEOF(__float128)
AC_CHECK_SIZEOF(double)

if [[ $ac_cv_sizeof___float128 != 16 ]] ; then
   AC_MSG_ERROR(__float128 is not 16 bytes)
fi			   

if [[ $ac_cv_sizeof_double != 8 ]] ; then
   AC_MSG_ERROR(double is not 8 bytes)
fi

export BITS=64
if [[ $ac_cv_sizeof_unsigned_long == 4 ]] ; then
   export BITS=32
   AC_MSG_WARN("running on 32-bit systems is slower")
fi

AC_PROG_CXX
AC_PROG_CC
# need to do something if these are not found
AC_PATH_PROG([MPICXX], [mpic++ mpicc], [])
AC_CHECK_FILE($MPICXX, [], [export HAVE_MPI=0; AC_DEFINE([HAVE_MPI], [0])]) 
#AC_CHECK_PROG([NVCC], [nvcc], [0], [export HAVE_CUDA=0])
AC_PATH_PROG([NVCC], [nvcc], [])
AC_PROG_MAKE_SET

# Checks for libraries.
# FIXME: Replace `main' with a function in `-lcudart':
AC_SEARCH_LIBS([cudaMalloc], [cudart], [], [export HAVE_CUDA=0; AC_DEFINE([HAVE_CUDA], [0])])
# FIXME: Replace `main' with a function in `-lgmp':
AC_SEARCH_LIBS(__gmpz_init, [gmp], [], 
[AC_SEARCH_LIBS(mpz_init, gmp, [], [AC_MSG_ERROR([libgmp not found])], [gmpxx])
			  , [gmpxx]])
# FIXME: Replace `main' with a function in `-lgmpxx':
# AC_CHECK_LIB([gmpxx], [main], [] [AC_MSG_ERROR([libgmpxx not found])])
# FIXME: Replace `main' with a function in `-lm':
AC_SEARCH_LIBS([sqrt], [m], [], [AC_MSG_ERROR([libmath not found])])
# FIXME: Replace `main' with a function in `-lmpfr':
AC_SEARCH_LIBS([mpfr_add, mpfr_init2], [mpfr], [], [AC_MSG_ERROR([libmpfr not found])])
# FIXME: Replace `main' with a function in `-lmpi':
#AC_SEARCH_LIBS([MPI_Finalize], [mpi], [], [export HAVE_MPI=0; AC_DEFINE([HAVE_MPI], [0])])
# FIXME: Replace `main' with a function in `-lprofiler':
#AC_CHECK_LIB([profiler], [main])
# FIXME: Replace `main' with a function in `-lpthread':
AC_SEARCH_LIBS([pthread_mutex_lock], [pthread], [], [AC_MSG_ERROR([libpthread not found])], [libpthread, thread])
#AX_PTHREAD( [
#            AC_DEFINE(HAVE_PTHREAD,1,[Define if you have POSIX threads libraries and header files.])
#            CLIBS="$PTHREAD_LIBS $LIBS"
#            CFLAGS="$CFLAGS $PTHREAD_CFLAGS"
#            LDFLAGS="$LDFLAGS $PTHREAD_CFLAGS"
#            CC="$PTHREAD_CC"],[AC_MSG_ERROR([libpthread not found])])
 
#Replace `main' with a function in `-lquadmath':
AC_SEARCH_LIBS([quadmath_snprintf], [quadmath], [], [AC_MSG_ERROR([libquadmath not found])])

# Checks for header files.
AC_CHECK_HEADERS([stdint.h stdlib.h unistd.h quadmath.h])
AC_CHECK_HEADERS([quadmath.h], [], AC_MSG_ERROR(no quadmath.h found))
AC_CHECK_HEADERS([cuComplex.h], [], [export HAVE_CUDA=0; AC_DEFINE([HAVE_CUDA], 0)])
#AC_MSG_CHECKING([for mpi.h])
#AC_TRY_COMPILE([#include <mpi.h>],[],[AC_MSG_RESULT(yes)], [export HAVE_MPI=0; AC_DEFINE([HAVE_MPI], [0]) AC_MSG_RESULT(no)])
#AC_CHECK_HEADERS([mpi.h], [], [export HAVE_MPI=0; AC_DEFINE([HAVE_MPI], 0)])
AC_CHECK_HEADERS([cuda_runtime.h], [], [export HAVE_CUDA=0; AC_DEFINE([HAVE_CUDA], 0)])

# the Hiary algorithm
# Should we add an option to revert to the Hiary algorithm if there are no gpu's in 
# the node? 

if [[ $HAVE_CUDA == 0 ]]; then
   if [[ $HAVE_MPI == 0 ]]; then
      AC_MSG_ERROR(No MPI or CUDA support ; compiling pointless as no advantages in using this version over the one already available on Bober's website)  
    fi
fi



#Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_INT32_T
AC_TYPE_SIZE_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_SUBST(HAVE_MPI)
AC_SUBST(HAVE_CUDA)
# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_FUNC_MALLOC
AC_CHECK_FUNCS([floor pow rint sqrt])
export ARCH=sm_30
AC_SUBST(ARCH)
AC_SUBST(BITS)
if [[ $HAVE_CUDA == 0 ]]; then
   AC_MSG_WARN(No CUDA detected ; this version of the program will be only usable as a stage3 node in a MPI cluster)
fi 

if [[ $HAVE_MPI == 0 ]]; then
   AC_MSG_WARN(No MPI detected ; this means that you will not be able to use the cluster version of this program)
fi

if [[ $HAVE_CUDA == 1 ]]; then
   AC_MSG_RESULT(Determining CUDA device capability...)
   nvcc -arch=sm_20 devicequery.cu
   ./a.out > /dev/null 2> /dev/null
   if [[ $? == 1 ]]; then
      AC_MSG_WARN(No CUDA device found ; setting compute capability to 3.0)
      export ARCH=sm_30
   else
      export ARCH=sm_`./a.out`    
      AC_MSG_RESULT(CUDA Compute Capability is $ARCH)
   fi
   rm a.out
fi   

AC_OUTPUT([Makefile])

AC_MSG_RESULT(Succesfully completed configuration)
