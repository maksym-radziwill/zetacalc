

ARCH=sm_37
ifneq (, )
CC = 
CXX = 
else
CC = gcc
CXX = g++
endif
BITS = 64

CXXFLAGS = -m$(BITS) -march=native -Wall -pthread -Winline -O3 -g -std=c++11 -Iinclude --param large-stack-frame=10000 --param large-stack-frame-growth=200 -ffast-math

H_CXXFLAGS = -m$(BITS) -march=native -Wall -pthread  -O3 -g -std=c++11 -Iinclude 

ifeq (1,1) 
NVCC = /usr/bin/nvcc
NVCCFLAGS = -arch=$(ARCH) -Xptxas=-v -O3 -D_FORCE_INLINES -Iinclude -Xcompiler -m64  -Xcompiler -fPIC
else
NVCC = g++ 
NVCCFLAGS = -Iinclude -x c 
endif

ifeq (0,1)
LDXXX = 
else	
LDXXX = g++
endif

ifeq (0, 1)
	ifeq (1, 1)
LDLIBS = -lmpfr -lgmp -lgmpxx -lpthread -lmpi -lcudart -lquadmath
	else
LDLIBS = -lmpfr -lgmp -lgmpxx -lpthread -lmpi -lquadmath
	endif
else
	ifeq (1, 1)
LDLIBS = -lmpfr -lgmp -lgmpxx -lpthread -lquadmath -lcudart
	else
LDLIBS = -lmpfr -lgmp -lgmpxx -lpthread -lquadmath
	endif
endif

THETA_SUM_OBJECTS = \
		    theta_sums/cache.o \
		    theta_sums/derivative_computations.o \
		    theta_sums/direct_evaluation.o \
		    theta_sums/exp_sum_euler_maclaurin.o \
		    theta_sums/G_functions.o \
		    theta_sums/H_and_J_integrals.o \
		    theta_sums/H_functions.o \
		    theta_sums/ICn.o \
		    theta_sums/misc.o \
		    theta_sums/theta_algorithm.o \
		    theta_sums/theta_sums.o \
		    theta_sums/w_coefficient.o


ifeq (1, 1)
MAIN_SUM_OBJECTS = \
		   main_sum/main_sum.o \
		   main_sum/ms_misc.o \
		   main_sum/stage1.o \
		   main_sum/stage2.o \
		   main_sum/stage2_gpu.o \
		   main_sum/stage3.o \
		   main_sum/md5.o
else
MAIN_SUM_OBJECTS = \
		   main_sum/main_sum.o \
		   main_sum/ms_misc.o \
		   main_sum/stage1.o \
		   main_sum/stage2.o \
		   main_sum/stage3.o \
	           main_sum/md5.o
endif

OTHER_OBJECTS = \
		log/log.o \
		misc/pow.o

OBJECTS = $(MAIN_SUM_OBJECTS) \
	  $(THETA_SUM_OBJECTS) \
	  $(OTHER_OBJECTS) \
	  zetacalc.o

EXECUTABLES = zetacalc

ALL_OBJECTS = $(MAIN_SUM_OBJECTS) \
		$(THETA_SUM_OBJECTS) \
		$(OTHER_OBJECTS) \
		zetacalc.o

zetacalc: $(ALL_OBJECTS)
	$(LDXXX)  $(ALL_OBJECTS) -o zetacalc $(LDLIBS)

theta_sums.a: $(THETA_SUM_OBJECTS) $(OTHER_OBJECTS)
	ar rs theta_sums.a $(THETA_SUM_OBJECTS) $(OTHER_OBJECTS)

ifeq (1,1)
main_sum/stage2_gpu.o : main_sum/stage2_gpu.cu
	$(NVCC) $(NVCCFLAGS) main_sum/stage2_gpu.cu -c -o main_sum/stage2_gpu.o 
endif

$(THETA_SUM_OBJECTS): include/theta_sums.h include/log.h include/misc.h theta_sums/precomputed_tables.h
$(MAIN_SUM_OBJECTS): include/theta_sums.h include/log.h include/main_sum.h include/misc.h theta_sums/precomputed_tables.h include/md5.h
log/log.o: include/log.h

theta_sums/H_functions.o: theta_sums/H_functions.cc
	$(CXX) -c theta_sums/H_functions.cc $(H_CXXFLAGS) -o theta_sums/H_functions.o

.PHONY: clean
clean:
	-rm $(OBJECTS)
	-rm $(EXECUTABLES)

cleanfiles: 
	-rm .file*

cleanconfig:
	-rm include/config.h
	-rm config.log
	-rm config.status
	-rm -rf autom4te.cache
	-rm Makefile

cleanold:
	-rm include/*~
	-rm theta_sums/*~
	-rm main_sum/*~
	-rm pow/*~
	-rm log/*~
	-rm ./*~

cleanall: clean cleanfiles cleanold cleanconfig 



