//
// MPI implementation of Dicod, a distributed algorithm to compute
// convolutional sparse coding, based on coordinate descent.
// See [Moreau2015] for details
//
// Author: Thomas Moreau (thomas.moreau@cmla.ens-cachan.fr)
// Date: May 2015
//



#include <mpi.h>
#include <chrono>
#include <thread>
#include "MPI_operations.h"
#include "worker.h"

using namespace MPI;
using namespace std;

int main(int argc, char**argv) {
    int * msg = new int[1];

	//Initiate the MPI API
	Init(argc, argv);
    cout << "Child Started" << endl;
	// Get the communicator and the Process API
	Intercomm parentComm = Comm::Get_parent();
    parentComm.Barrier();
    cout << "Bc1" << endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    parentComm.Barrier();
    cout << "Bc2" << endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    parentComm.Barrier();
    cout << "Receiving" << endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    parentComm.Recv(msg, 1, INT, 0, 100);
    cout << "Stopping msg" << endl;
    parentComm.Barrier();

    Finalize();
}
