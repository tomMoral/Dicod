//
// MPI operations refactored
//
// Author: Thomas Moreau  (thomas.moreau@cmla.ens-cachan.fr)
// Date: May 2015
//
#include <mpi.h>
using namespace MPI;

double* receive_bcast(Intercomm* comm);
double* receive_bcast(Intercomm* comm, int &size);
void confirm_array(Intercomm* comm, double a0, double a1);
int clean_up(Intercomm* comm, bool debug, int rank);