//
// MPI implementation of Dicod, a distributed algorithm to compute
// convolutional sparse coding, based on coordinate descent.
// See [Moreau2015] for details
//
// Author: Thomas Moreau (thomas.moreau@cmla.ens-cachan.fr)
// Date: May 2015
//



#include <mpi.h>
#include "MPI_operations.h"
#include "worker.h"

using namespace MPI;
using namespace std;

int main(int argc, char**argv) {

	//Initiate the MPI API
	Init(argc, argv);

	// Get the communicator and the Process API
	Intercomm parentComm = Comm::Get_parent();
	Worker *worker = new Worker(&parentComm);
	worker->start();
	int rank = worker->getRank();
	delete worker;

    return clean_up(&parentComm, DEBUG, rank);
}
