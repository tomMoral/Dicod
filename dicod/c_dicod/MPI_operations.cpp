//
// MPI operations refactored
//
// Author: Thomas Moreau  (thomas.moreau@cmla.ens-cachan.fr)
// Date: May 2015
//

#include <iostream>
#include "MPI_operations.h"

using namespace std;


double* receive_bcast(Intercomm* comm){
	int tmp;
	return receive_bcast(comm, tmp);
}

double* receive_bcast(Intercomm* comm, int &size){

	int* s = &size;
	comm->Bcast(s, 1, INT, 0);

	double* out = new double[size];
	comm->Bcast(out, size, DOUBLE, ROOT);
	return out;
}

void confirm_array(Intercomm* comm, double a0, double a1){
	double confirm[2];
	confirm[0] = a0;
	confirm[1] = a1;
	comm->Gather(confirm, 2, DOUBLE, NULL, NULL_SIZE, DOUBLE, ROOT);
}

int clean_up(Intercomm* comm, bool debug, int rank){
	if(debug && rank == 0)
		cout << "DEBUG  - MPI - clean end" << endl;
	comm->Disconnect();
	//comm->Free();
	Finalize();
	if(Is_finalized())
		return 0;
	return 1;
}
