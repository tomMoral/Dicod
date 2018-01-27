
#ifndef DICOD_H
#define DICOD_H

#include <mpi.h>
#include <time.h>
#include <unordered_map>
#include <list>
#include <chrono>
#include <thread>
#include <random>

//Define messages info
#define STOP 0
#define UP 1
#define REQ_PROBE 2
#define REP_PROBE 3

// Enum algo coeff selection
#define ALGO_GS 0
#define ALGO_RANDOM 1

// Smallest coefficient update for ALGO_RANDOM
#define EPSILON 1e-10

// Message constants
#define HEADER 7
#define TAG_UP 2742

using namespace MPI;
using namespace std;


class DICOD
{
	public:
		//Construction and destruction
		DICOD(Intercomm* _parentComm);
		~DICOD();

		//Manage the process
		double step();
		void start();
		void end();
		bool stop(double dz);
		int get_rank(){ return world_rank;}
		bool get_dbg(){ return debug;}
		const char* is_paused(){ return (pause)?"paused":"not paused";}
		void reduce_pt();
		void receive_task();

	private:
		//Private Attributes
		Intercomm *parentComm;

		double *sig, *beta, *pt; // Signal, beta
		double *alpha_k, *DD, *D;
		bool *end_neigh, first_probe;
		double lmbd, tol, t_max;
		long int iter, i_max;
		int L_proc, L_proc_S, proc_off;
		int T, dim, S, K, L;
		int world_size, world_rank;
		int algo, patience, max_probe;
		double next_probe, up_probe, runtime, t_init;
		chrono::high_resolution_clock::time_point t_start;
		bool pause, go, debug, logging, positive;
		list<double*> messages;
		unordered_map<int, int> probe_result;
		list<int> probe_try;
		list<double> log_dz, log_time, log_i0, log_skip;
		mt19937 rng;

		// Segment routine variables
		int seg_size;
		int current_seg, n_seg, n_zero;
		double seg_dz;

		// Random algorithm count
		int n_skip;

		//Private Methods
		double _return_dz(double dz);
		double compute_cost();
		double _check_convergence();
		void _init_algo();
		void _update_beta(double dz, int k, int t);
		void process_queue();
		void send_update_msg(int dest, double dz, int k0, int cod_start, int DD_start, int ll);
		void send_msg(int msg_type, int arg, bool up);
		void Ibroadcast(int msg_t);
		void probe_reply();

};

/*
void solve_DICOD(Intercomm *parentComm, int& rank, bool& debug){
	double dz = 100.;
	DICOD *dcp = new DICOD(parentComm);
	while(!dcp->stop(dz))
		dz = dcp->step();

	// Send back
	dcp->reduce_pt();
	dcp->end();
    debug = dcp->get_dbg();
    rank = dcp->get_rank();
}*/

#endif
