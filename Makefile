# On Euler, we will benchmark your DGEMM's performance against the performance
# of the default vendor-tuned DGEMM. This is done in benchmark-blas.
#

CC = gcc
OPT = -O2
CFLAGS = -Wall -std=gnu99 $(OPT)
LDFLAGS = -Wall
# librt is needed for clock_gettime
LDLIBS = -lrt -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl -m64 -I${MKLROOT}/include

targets = benchmark-naive benchmark-naive-transpose benchmark-blocked-simple benchmark-blocked-simple-transpose benchmark-blocked-unroll benchmark-blocked-unroll-transpose benchmark-blas
objects = benchmark.o dgemm-naive.o dgemm-naive-transpose.o dgemm-blocked-simple.o dgemm-blocked-simple-transpose.o dgemm-blocked-unroll.o dgemm-blocked-unroll-transpose.o dgemm-blas.o  

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-naive-transpose : benchmark.o dgemm-naive-transpose.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-simple : benchmark.o dgemm-blocked-simple.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-simple-transpose : benchmark.o dgemm-blocked-simple-transpose.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-unroll : benchmark.o dgemm-blocked-unroll.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-unroll-transpose : benchmark.o dgemm-blocked-unroll-transpose.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
