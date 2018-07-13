
EXECS=c_dicod start_worker _test_send
MPICC?=mpic++
STD11=-std=c++11
FFTW=`pkg-config --libs --cflags fftw3`
OPTIONFLAGS=${STD11}

all: ${EXECS} clean_bld

c_dicod: c_dicod.cpp dicod.o MPI_op.o fftw_conv.o
	${MPICC} ${OPTIONFLAGS} -o c_dicod c_dicod.cpp dicod.o MPI_op.o fftw_conv.o ${FFTW}

start_worker: start_worker.cpp worker.o MPI_op.o fftw_conv.o dicod.o dicod2d.o
	${MPICC} ${OPTIONFLAGS} -o start_worker start_worker.cpp MPI_op.o fftw_conv.o worker.o dicod.o dicod2d.o ${FFTW}

test_barriere: test_barriere.cpp
	${MPICC} ${OPTIONFLAGS} -o test_barriere test_barriere.cpp

_test_send: _test_send.cpp test_barriere
	${MPICC} ${OPTIONFLAGS} -o _test_send _test_send.cpp

worker.o: worker.cpp worker.h dicod.o dicod2d.o
	${MPICC} ${OPTIONFLAGS} -c -o worker.o worker.cpp

dicod.o: dicod.cpp dicod.h  MPI_op.o
	${MPICC} ${OPTIONFLAGS} -c -o dicod.o dicod.cpp

dicod2d.o: dicod2d.cpp dicod2d.h constants.h MPI_op.o
	${MPICC} ${OPTIONFLAGS} -c -o dicod2d.o dicod2d.cpp

MPI_op.o: MPI_operations.cpp MPI_operations.h constants.h
	${MPICC} -c -o MPI_op.o MPI_operations.cpp

fftw_conv.o: convolution_fftw.c convolution_fftw.h
	${MPICC} ${OPTIONFLAGS} -c -o fftw_conv.o convolution_fftw.c

clean: clean_bld
	rm ${EXECS} *.o

clean_bld:
	rm *.o
