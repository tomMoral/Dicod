
#ifndef DICOD2D_H
#define DICOD2D_H

#include <mpi.h>
#include <time.h>
#include <unordered_map>
#include <list>
#include <chrono>
#include <thread>
#include <random>
#include "constants.h"

using namespace MPI;
using namespace std;

typedef chrono::high_resolution_clock::time_point time_point;
typedef chrono::duration<double> d_duration;


class DICOD2D
{
	public:
		//Construction and destruction
		DICOD2D(Intercomm* _parentComm);
		~DICOD2D();

		//Manage the process
		void start();
		double step();
		bool stop(double dz);
		void send_result();
		void end();

		// Properties
		int get_rank(){ return world_rank;}
		bool get_dbg(){ return debug;}
		const char* is_paused(){ return (pause)?"paused":"not paused";}

	private:
		//Private Attributes
		Intercomm *parentComm;			// Communicator for MPI operations

		double *sig, *beta, *pt; 		// Signal, beta and current point
		double *D, *alpha_k, *DD;		// Dicitonary, norm of the dict and cross correlation

		// Algorithm parameters
		double lmbd, tol, t_max;
		int i_max, n_seg, algo, patience;
		bool debug, logging, positive;

		// dimension of the problem
		int dim, K, h_dic, w_dic, S;	// Dimensions of the dictionary
		int h_cod, w_cod, L;			// Dimensions of the weights
		int h_sig, w_sig;				// Dimensions of the signal
		int h_world, w_world, world_size;
										// Dimensions of the processor grid

		// dimension of the problem on this processor
		int h_proc, w_proc, L_proc;		// Dimensions of the weights
		int h_proc_S, w_proc_S;			// Dimensions of the signal
		int h_off, w_off;				// Offsets of the processor
		int h_rank, w_rank, world_rank;	// Position on the processor grid

		// Running properties
		int iter;						// Number of iteration
		int  n_barrier;					// Number of stopped processes
		int n_zero;						// Number of consecutive non-update steps
		bool *end_neigh;				// Hold the state of the 8 neighbors
		time_point t_start;				// Hold the starting time of the algorithm to compute the runtime
		double t_init, runtime;			// Runtime for initialization & convergence
		bool pause, go;					// State of the processor
		int seg_size, current_seg;		// Size of each segment for the segmented algo and current segment index


		mt19937 rng;					// Random number generator for the random cooridnate choice
		list<double*> messages;			// List all the sent messages
		double next_probe, up_probe;	// Hold the information about the next probing time
		list<int> probe_try;			// Hold the received probe requests
		unordered_map<int, int> probe_result;
										// Hold the number of processes that reply to a given probe
		list<double> log_dz, log_t, log_i0;
										// List that hold all the update time and value


    	//Private Methods
		void _rcv_task();
		void _init_algo();
		double _choose_coord_GS(int, int, int&, int&);
		double _choose_coord_Rand(int, int, int&, int&);
    	double _return_dz(double dz);
		double compute_cost();
		void _update_beta(double dz, int k, int t, int h, int w);
		void process_queue();
		void _signal_end();
		void _send_msg(int dest, int msg_type, int arg = 0);
		void Ibroadcast(int msg_t);
		void probe_reply();
		double _get_time_span();
		void send_updates(double dz, int k0, int w0, int h0,
						  int h_cod_start, int h_dic_start, int h_ll, 
						  int w_cod_start, int w_dic_start, int w_ll);
		void send_update_msg(int dest, double dz, int k0,
							 int h_cod_start, int h_dic_start, int h_ll,
							 int w_cod_start, int w_dic_start, int w_ll);
	
	
};


void solve_DICOD(Intercomm *parentComm);

#endif
