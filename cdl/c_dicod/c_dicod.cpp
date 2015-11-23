//
// MPI implementation of Dicod, a distributed algorithm to compute
// convolutional sparse coding, based on coordinate descent.
// See [Moreau2015] for details
//
// Author: Thomas Moreau (thomas.moreau@cmla.ens-cachan.fr)
// Date: May 2015
//



#include <mpi.h>
#include <math.h>
#include "dicod.h"
#include "MPI_operations.h"

int main(int argc, char**argv) {

	//Initiate the MPI API
	Init(argc, argv);

	// Get the communicator and the Process API
	Intercomm parentComm = Comm::Get_parent();
	double dz = 100.;
	DICOD *dcp = new DICOD(&parentComm);
	while(!dcp->stop(dz))
		dz = dcp->step();

	// Send back
	dcp->reduce_pt();
	dcp->end();
    bool debug = dcp->get_dbg();
    int rank = dcp->get_rank();
	free(dcp);
    return clean_up(&parentComm, debug, rank);
}
