
EXECS=c_dicod start_worker
MPICC?=mpic++
STD11=-std=c++11
FFTW=`pkg-config --libs --cflags fftw3`
OPTIONFLAGS=${STD11}

all: ${EXECS} clean_bld

c_dicod: c_dicod.cpp dicod.o MPI_op.o fftw_conv.o
	${MPICC} ${OPTIONFLAGS} -o c_dicod c_dicod.cpp dicod.o MPI_op.o fftw_conv.o ${FFTW}

start_worker: start_worker.cpp worker.o MPI_op.o dicod.o
	${MPICC} ${OPTIONFLAGS} -o start_worker start_worker.cpp MPI_op.o worker.o dicod.o ${FFTW}

worker.o: worker.cpp worker.h dicod.o
	${MPICC} ${OPTIONFLAGS} -c -o worker.o worker.cpp

dicod.o: dicod.cpp dicod.h  MPI_op.o
	${MPICC} ${OPTIONFLAGS} -c -o dicod.o dicod.cpp

MPI_op.o: MPI_operations.cpp MPI_operations.h
	${MPICC} -c -o MPI_op.o MPI_operations.cpp

fftw_conv.o: convolution_fftw.c convolution_fftw.h
	${MPICC} ${OPTIONFLAGS} -c -o fftw_conv.o convolution_fftw.c

clean: clean_bld
	rm ${EXECS} *.o

clean_bld:
	rm *.o
