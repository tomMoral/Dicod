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
	double dz = 100;
	dz = 100.;
	DICOD *pc = new DICOD(&parentComm);
	while(!pc->stop(dz)){
		dz = pc->step();
	}

	// Send back
	pc->reduce_pt();
	pc->end();
    bool debug = pc->get_dbg();
    int rank = pc->get_rank();
	free(pc);
    return clean_up(&parentComm, debug, rank);
}
