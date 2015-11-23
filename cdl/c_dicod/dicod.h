#include <mpi.h>
#include <time.h>
#include <unordered_map>
#include <list>
#include <chrono>
#include <thread>

//Define messages info
#define STOP 0
#define PAUSE 1
#define UP 2
#define REQ_PROBE 3
#define REP_PROBE 4

#define HEADER 7

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
		int L_proc, L_proc_S, off, iter;
		int T, dim, S, K, L, i_max;
		int world_size, world_rank;
		double next_probe, up_probe, runtime, t_init;
		chrono::high_resolution_clock::time_point t_start;
		bool pause, go, debug, logging, positive;
		list<double*> messages;
		unordered_map<int, int> probe_result;
		list<int> probe_try;
		list<double> log_dz, log_t, log_i0;

		// Segment routine variables
		int use_seg, seg_size;
		int current_seg, n_seg, n_zero;
		double seg_dz;

    	//Private Methods
    	double _return_dz(double dz);
		double compute_cost();
		void _init_algo();
		void _update_beta(double dz, int k, int t);
		void process_queue();
		void send_update(double dz, int ll, int off, int k0);
		void send_msg(int msg_type, int arg, bool up);
		void Ibroadcast(int msg_t);
		void probe_reply();
	
};
