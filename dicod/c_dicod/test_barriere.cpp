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
    int * msg = new int[1];
    const char* program = "dicod/c_dicod/_test_send";
    msg[0] = 42;

	//Initiate the MPI API
	Init(argc, argv);
    cout << "Parent Started" << endl;

    Info info = Info::Create();
    info.Set("add-hostfile", "hostfile");


    Intercomm comm = COMM_WORLD.Spawn(program, NULL, 1, info, 0);

	// Get the communicator and the Process API
    comm.Barrier();
    cout << "Bp1" << endl;
    comm.Barrier();
    cout << "Bp2" << endl;
    comm.Barrier();
    cout << "Sending" << endl;
    comm.Send(msg, 1, INT, 0, 100);
    cout << "Sent" << endl;
    comm.Barrier();

    Finalize();
}
