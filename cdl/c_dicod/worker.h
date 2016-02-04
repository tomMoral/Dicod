#ifndef WORKER_MPI
#define WORKER_MPI
#include <iostream>
#include <chrono>
#include <thread>
#include <mpi.h>

#define DEBUG true

//CONTROL MSG
#define STOP 0
#define RESIZE_SERVER 1
#define RESIZE_CLIENT 2
#define SOLVE_DICOD 3
#define SOLVE_DICOD2D 4

// CONTROL MSG TAG
#define TAG_MNG_MSG 0
#define TAG_PORT_MSG 1

#define RUN 1

using namespace MPI;
using namespace std;

class Worker
{

private:
	Intercomm* parentComm;
	int world_size, world_rank;
	void control_loop();
public:
	Worker(Intercomm*);
	~Worker();
	void start();
	int getRank(){ return world_rank;}
	
};

#endif