//
// mPI implementation of Dicod, a distributed algorithm to compute
// convolutional sparse coding, based on coordinate descent.
// see [Moreau2015] for details
//
// author: Thomas Moreau  (thomas.moreau@cmla.ens-cachan.fr)
// date: February 2016
//
#include "dicod2d.h"

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "convolution_fftw.h"
#include "MPI_operations.h"
using namespace FFTW_Convolution;


#define PAUSE_DELAY 3			// sleep time for paused process in ms
#define PROBE_MSG_UP_START 0.05		// smallest time between probe for end
#define PROBE_MAX 0.4			// largest time between probes

#define DEBUG false


// solve a problem in 2D with DICOD algorithm
void solve_DICOD(Intercomm *parentComm){
	double dz = 100.;
	DICOD2D *dcp = new DICOD2D(parentComm);
	dcp->start();
	while(!dcp->stop(dz))
		dz = dcp->step();

	// send back
	dcp->send_result();
	dcp->end();
}

//Object to handle computations
DICOD2D::DICOD2D(Intercomm* _parentComm){
	// init time measurement for initialization
	t_start = chrono::high_resolution_clock::now();

	parentComm = _parentComm;

	// initiate arrays
	alpha_k = NULL, DD=NULL, D=NULL;
	sig = NULL, beta = NULL, pt=NULL;
	runtime = 0;

	// greetings
	world_size = parentComm->Get_size();	// # processus
	world_rank = parentComm->Get_rank();	// rank in the processus pool
	if(DEBUG){
		// get the machine this process is running on
		char* procName = new char[MAX_PROCESSOR_NAME];
		int plen;
		Get_processor_name(procName, plen);
		cout << "DEBUG - MPI_Worker" << world_rank << " - Start processor "
			 << world_rank << "/" << world_size
			 << " on " << procName << endl
			 << "    and with COMM_WORLD " << COMM_WORLD.Get_rank() << "/"
			 << COMM_WORLD.Get_size() << endl;
		delete[] procName;
	}
}

// destructor, delete all arrays
DICOD2D::~DICOD2D(){
	delete[] D;
	delete[] DD;
	delete[] alpha_k;
	delete[] sig;
	delete[] pt;
	delete[] beta;
	delete[] end_neigh;
}

void DICOD2D::start(){
	int seed =  chrono::duration_cast<chrono::milliseconds>(
                   t_start.time_since_epoch()).count();
	rng.seed(world_rank*seed);
	_rcv_task();


	// init algo and wait for everyone
	_init_algo();
	parentComm->Barrier();

	// get a timer to evaluate the performances
	t_init = _get_time_span();
  	if((debug || DEBUG) && world_rank == 0)
  		cout << "DEBUG - MPI_Workers- End initialization in " << t_init << endl;
	t_start = chrono::high_resolution_clock::now();
}

// handle initial communication
void DICOD2D::_rcv_task(){

	// update dictionary constants
	delete[] alpha_k;
	delete[] DD;
	delete[] D;
	alpha_k = receive_bcast(parentComm);
	DD = receive_bcast(parentComm);
	D = receive_bcast(parentComm);

	//Receives some constant of the algorithm
	double* constants = receive_bcast(parentComm);
	dim = (int) constants[0];				// dimension of the signal
	K = (int) constants[1];					// number of dictionary elements
	// sizes of the dicitonary S
	h_dic = (int) constants[2];
	w_dic = (int) constants[3];
	S = h_dic*w_dic;
	// sizes of the signal T
	h_sig = (int) constants[4];
	w_sig = (int) constants[5];

	// topology of the processor grid
	w_world = (int) constants[6];			// number of proc horizontally
	h_world = world_size / w_world;

	// compute topology position
	w_rank = world_rank % w_world;
	h_rank = world_rank / w_world;

	lmbd = constants[7];					// regularisation parameter
	tol = constants[8];						// convergence tolerance
	t_max = constants[9];					// maximum time
	i_max = (int) constants[10];			// # iterations maximum
	debug = ((int) constants[11] > 0);		// debug level
	logging = ((int) constants[12] == 1);	// activate the logging
	n_seg = ((int) constants[13]);		// use a semgneted update
	positive = ((int) constants[14] == 1);	// use to only activate positive updates
	algo =(int) constants[15];				// coordinate choice algorihtm
	patience = (int) constants[16];			// max number of 0 updates in ALGO_RANDOM
	delete[] constants;

	if(algo == ALGO_GS)
		patience = 1;
	patience *= n_seg;

	if(world_rank == 0 && (DEBUG || debug))
		cout << "DEBUG - MPI_Workers- Start with algorithm : "
			 << ((ALGO_GS==algo)?"Gauss-Southwell":"Random") << endl;

	// total sizes of the code
	h_cod = h_sig-h_dic+1;   // height of the code per proc
	w_cod = w_sig-w_dic+1;   // width of the code per proc
	L = h_cod*w_cod;

	// sizes of the code computed by the processor
	h_proc = h_cod / h_world;
	w_proc = w_cod / w_world + 1;

	// offset of the position of the processor
	h_off = h_rank*h_proc;
	w_off = w_rank*w_proc;

	h_proc = min(h_off+h_proc, h_cod)-h_off;
	w_proc = min(w_off+w_proc, w_cod)-w_off;
	L_proc = h_proc*w_proc;

	// size of the signal received by the proc
	h_proc_S = h_proc+h_dic-1;
	w_proc_S = w_proc+w_dic-1;

	// receive the signal to process
	delete[] sig;
	sig = new double[dim*h_proc_S*w_proc_S];
	parentComm->Recv(sig, dim*h_proc_S*w_proc_S, DOUBLE, ROOT,
					 TAG_MSG_ROOT+world_rank);

	confirm_array(parentComm, sig[0], sig[dim*h_proc_S*w_proc_S-1]);
}

// init the algo
void DICOD2D::_init_algo(){
	int k, t, d, tau;
	double bh;
	double* src, *kernel;
	int L_rec = h_proc_S*w_proc_S;

	// fFT tools
	Workspace ws;
	init_workspace(ws, LINEAR_VALID, h_proc_S, w_proc_S, h_dic, w_dic);

 	// the main loop

	//Initialize beta and pt
	delete[] beta;
	delete[] pt;
	beta = new double[K*L_proc];
	pt = new double[K*L_proc];
	fill(beta, beta+K*L_proc, 0);
	fill(pt, pt+K*L_proc, 0);
	kernel = new double[S];
	for(k = 0; k < K; k++){
		for(d=0; d < dim; d++){
			src = sig+d*L_rec;
			//Revert dic in both direction
			for(tau=0; tau < S; tau++)
				kernel[tau] = D[(k*dim+(d+1))*S-tau-1];
  			convolve(ws, src, kernel);
  			for(t=0; t < L_proc; t++){
  			 	beta[k*L_proc+t] -= ws.dst[t]/dim;
  			}
		}
	}
	clear_workspace(ws);
	delete[] kernel;

	iter = 0;
	pause = false;
	go = true;
	next_probe = 0;
	up_probe = PROBE_MSG_UP_START;
	log_dz.clear();
	log_i0.clear();
	log_t.clear();

	// init the segment choosing and stoping
	current_seg = 0;
	seg_size = ceil(L_proc * 1. / n_seg);
	n_zero = 0;
	n_barrier = 0;

	end_neigh = new bool[8];
	end_neigh[0] = !(w_rank > 0);
	end_neigh[1] = !(w_rank > 0 && h_rank > 0);
	end_neigh[2] = !(h_rank > 0);
	end_neigh[3] = !(w_rank < w_world-1 && h_rank > 0);
	end_neigh[4] = !(w_rank < w_world-1);
	end_neigh[5] = !(w_rank < w_world-1 && h_rank < h_world-1);
	end_neigh[6] = !(h_rank < h_world-1);
	end_neigh[7] = !(w_rank > 0 && h_rank < h_world-1);
}

// choose coordinate to update with Gauss-southwell rule
double DICOD2D::_choose_coord_GS(int seg_start, int seg_end,
								 int &k0, int &t0)
{
	//Find argmax of |z_i - z'_i|
	int k, k_off = 0, t, i;
	double beta_i, sign_beta_i, ak, dz, adz = tol;
	for(k = 0; k < K; k++){
		ak = alpha_k[k];
		for (t=seg_start; t < seg_end; t++){
			i = k_off+t;
			beta_i = -beta[i];
			sign_beta_i = (beta_i >= 0)?1:((positive)?0:-1);
			beta_i = max(0., fabs(beta_i)-lmbd)*sign_beta_i/ak;
			if(adz < fabs(beta_i-pt[i])){
				k0 = k;
				t0 = t;
				dz = pt[i]-beta_i;
				adz = fabs(dz);
			}
		}
		k_off += L_proc;
	}
	return dz;
}

// choose coordinate to update with Gauss-southwell rule
double DICOD2D::_choose_coord_Rand(int seg_start, int seg_end,
								   int &k0, int &t0)
{

	int i, k, t;
	double beta_i, sign_beta_i, ak, dz;
	uniform_int_distribution<> dis_k(0, K-1), dis_t(seg_start, seg_end-1);

	// draw a random coordinate uniformely
	t = dis_t(rng);
	k  = dis_k(rng);

	// compute the update
	ak = alpha_k[k];
	i = k*L_proc+t;
	beta_i = -beta[i];
	sign_beta_i = (beta_i >= 0)?1:-1;
	if(positive)
		sign_beta_i = (beta_i >= 0)?1:0;
	beta_i = max(0., fabs(beta_i)-lmbd)*sign_beta_i/ak;

	if(tol < fabs(beta_i-pt[i])){
		k0 = k;
		t0 = t;
		dz = pt[i]-beta_i;
	}
	return dz;
}

// on step of the coordinate descent
double DICOD2D::step(){
	if(pause)
		this_thread::sleep_for(chrono::milliseconds(PAUSE_DELAY));

	process_queue();
	//int i, k, t, k_off;
	int k0, w0, h0, t0 = -1;
	double dz, adz;
	if(pause){
		if(probe_try.size()>0)
			probe_reply();
		return tol;
	}
	probe_try.clear();

	// compute the current segment if we use the
	// segmented version of the algorithm
	int seg_start = 0;
	int seg_end = L_proc;
	if(n_seg > 1){
		seg_start = current_seg*seg_size;
		seg_end = (current_seg+1)*seg_size;
		current_seg += 1;
		if(seg_end >= L_proc)
			current_seg = 0;
		seg_end = min(seg_end, L_proc);
	}

	// choose a coordinate to update
	if(algo == ALGO_GS)
		dz = _choose_coord_GS(seg_start, seg_end, k0, t0);
	else if(algo == ALGO_RANDOM)
		dz = _choose_coord_Rand(seg_start, seg_end, k0, t0);
	adz = fabs(dz);

	// if the update is too small, do not update and return a 0 update
	if(adz <= tol)
		return _return_dz(adz);;

	// else perform the update and reset the consecutive zero counter
	pt[k0*L_proc+t0] -= dz;
	iter += 1;
	n_zero = 0;

	// log the update if the logging is active
	h0 = t0 / w_proc;
	w0 = t0 % w_proc;
	if(logging){
	  	double seconds = _get_time_span();

		log_t.push_back(seconds);
		log_dz.push_back(-dz);
		log_i0.push_back((double) k0*L+(h_off+h0)*w_cod+w_off+w0);
	}

	// update beta
	_update_beta(dz, k0, t0, h0, w0);

	return _return_dz(adz);
}

// handle the computation to avoid ending computation before convergence in
// random algorithm and in segmented iterations
double DICOD2D::_return_dz(double dz){
	if(dz <= tol)
		n_zero += 1;
	if(n_zero < patience)
		dz = 2*tol;
	return dz;
}

// update the beta variable after a change of coordinate (k0, t0) of dz
void DICOD2D::_update_beta(double dz, int k0, int t0, int h0, int w0){
	//Loop variables
	int k, h_tau, w_tau;

	//Offset variables
	int DD_off, beta_off;
	int s_DD= (2*h_dic-1)*(2*w_dic-1);
	int h_DD_start, w_DD_start, h_cod_start, w_cod_start, h_ll, w_ll;

	//Hold previous beta for the current indice
	int i0 = k0*L_proc+t0;
	double p_beta_i = beta[i0];

	// compute offsets
	h_DD_start = max(0, h_dic-h0-1);
	w_DD_start = max(0, w_dic-w0-1);
	h_cod_start = max(0, h0-h_dic+1);
	w_cod_start = max(0, w0-w_dic+1);
	h_ll = min(h_proc, h0+h_dic) - h_cod_start;
	w_ll = min(w_proc, w0+w_dic) - w_cod_start;

	// update beta localy
	for(k=0; k < K; k++){
		beta_off = k*L_proc + h_cod_start*w_proc + w_cod_start;
		DD_off = k*K*s_DD + k0*s_DD + h_DD_start*(2*w_dic-1) + w_DD_start;
		for(h_tau=0; h_tau < h_ll; h_tau++){
			for(w_tau=0; w_tau < w_ll; w_tau++)
				beta[beta_off+w_tau] -= DD[DD_off+w_tau]*dz;
			beta_off += w_proc;
			DD_off += (2*w_dic-1);
		}
	}

	// set beta to its previous value as it should not be updated
	beta[i0] = p_beta_i;

	// send messages to neighboors
	send_updates(dz, k0, w0, h0, h_cod_start, h_DD_start, h_ll,
				 w_cod_start, w_DD_start, w_ll);
}

// send updates messages to neighbors
void DICOD2D::send_updates(double dz, int k0, int w0, int h0,
						   int h_cod_start, int h_DD_start, int h_ll,
						   int w_cod_start, int w_DD_start, int w_ll)
{
	if(w0 < w_dic && w_rank > 0)
			send_update_msg(world_rank - 1, dz, k0,
							h_cod_start, h_DD_start, h_ll,		// left neighbor
							-w_DD_start, 0, w_DD_start);

	if(w0 > w_proc-w_dic && w_rank < w_world - 1)
			send_update_msg(world_rank + 1, dz, k0,
							h_cod_start, h_DD_start, h_ll,		// right neighbor
							0, w_ll, (2*w_dic-1)-w_ll);

	if(h0 < h_dic && h_rank > 0){
		if(w0 > w_proc-w_dic && w_rank < w_world - 1)
			send_update_msg(world_rank - w_world + 1, dz, k0,
				 			-h_DD_start, 0, h_DD_start,			// uper-right neighbor
							0, w_ll, (2*w_dic-1)-w_ll);

			send_update_msg(world_rank - w_world, dz, k0,
							-h_DD_start, 0, h_DD_start,			// upper neighbor
							w_cod_start, w_DD_start, w_ll);

		if(w0 < w_dic && w_rank > 0)
			send_update_msg(world_rank - w_world - 1, dz, k0,
							-h_DD_start, 0, h_DD_start,			// upper-left neighbor
							-w_DD_start, 0, w_DD_start);
	}

	else if(h0 > h_proc-h_dic && h_rank < h_world - 1){
		if(w0 > w_proc-w_dic && w_rank < w_world - 1)
			send_update_msg(world_rank + w_world + 1, dz, k0,
							0, h_ll, (2*h_dic-1)-h_ll,			// lower-right neightbor
							0, w_ll, (2*w_dic-1)-w_ll);

			send_update_msg(world_rank + w_world, dz, k0,
							0, h_ll, (2*h_dic-1)-h_ll,			// lower neighbor
							w_cod_start, w_DD_start, w_ll);

		if(w0 < w_dic && w_rank > 0)
			send_update_msg(world_rank + w_world - 1, dz, k0,
							0, h_ll, (2*h_dic-1)-h_ll,			// lower-left neighbor
							-w_DD_start, 0, w_DD_start);
	}
}

// send update message to th neighbor dest
void DICOD2D::send_update_msg(int dest, double dz, int k0,
							  int h_cod_start, int h_DD_start, int h_ll,
							  int w_cod_start, int w_DD_start, int w_ll)
{
	// construct the message
 	double* msg = new double[HEADER_2D];
	msg[0] = (double) MSG_UP;
	msg[1] = dz;
	msg[2] = (double) k0;
	msg[3] = (double) h_cod_start;
	msg[4] = (double) w_cod_start;
	msg[5] = (double) h_DD_start*(2*w_dic-1)+w_DD_start;
	msg[6] = (double) h_ll;
	msg[7] = (double) w_ll;
	messages.push_back(msg);

	// send the message
	COMM_WORLD.Isend(msg, HEADER_2D, DOUBLE,
				     dest, TAG_MSG_UP);

	// fail all previous probes as we will wake at least one process with this msg
	unordered_map<int,int>::iterator it;
	for(it=probe_result.begin(); it != probe_result.end(); it++)
		it->second --;
}

bool DICOD2D::stop(double dz){
	// compute the current runtime
	double seconds = _get_time_span();

  	// if we have reach an optimal solution, stop the algorithm
	if(!go){
		COMM_WORLD.Barrier();
		if(world_rank == 0 && (debug || DEBUG))
			cout << "INFO - MPI_Workers- Reach optimal solution in " << seconds
				 << endl;
		return true;
	}

	// print debug message to notify that we reach a timeout or the maximal iteration
	if((debug || DEBUG) && seconds >= t_max && world_rank == 0)
		cout << "DEBUG - MPI_Workers- Reach timeout" << endl;
	if((debug || DEBUG) && iter >= i_max && world_rank == 0)
		cout << "DEBUG - MPI_Workers- Reach max iteration" << endl;

	// if we have reach an optimal solution within the process
	// enter pause state if the other processes are still running
	if(fabs(dz) <= tol){
		// the process with rank 0 will coordinate the ending
		if(world_rank == 0){
			// if it was the last one running, stop the algorithm
			if(world_size - n_barrier == 1){
				go = false;
				pause = true;
				runtime = seconds;
				COMM_WORLD.Barrier();
				return true;
			}
			// else if it just enter the pause state, initiate the probe process
			if(!pause){
				cout << "DEBUG - MPI_Worker" << world_rank << "Enter pause" << endl;
				up_probe = PROBE_MSG_UP_START;
				next_probe = seconds;
				pause = true;
				runtime = seconds;
			}
			// probe the other process to know if they are still running
			if(next_probe <= seconds){
				Ibroadcast(MSG_REQ_PROBE);
				up_probe = max(up_probe*1.2, PROBE_MAX);
				next_probe = seconds + up_probe;
			}
		}
		// for the other process
		else if(!pause){
			if(debug || DEBUG)
			cout << "DEBUG - MPI_Worker" << world_rank << " - Enter pause" << endl;
			pause = true;
			runtime = seconds;
		}
	}
	bool _stop = false;
	_stop |= (iter >= i_max);
	_stop |= (seconds >= t_max);
	if(_stop){
		runtime = seconds;
		cout << "DEBUG - MPI_Worker" << world_rank << " - Enter barrier as it "
			 << "reached t_max/i_max" << endl;
		double *msg = NULL;
		if(world_rank != 0){
			_send_msg(0, MSG_HIT_BARRIER);
		}
		else{
			up_probe = PROBE_MSG_UP_START;
			next_probe = seconds;
			cout << "DEBUG - MPI_Worker0 - will wait until other process end "
				 << "or paused" << endl;
			while(go && n_barrier < world_size-1){
				pause = true;
				this_thread::sleep_for(chrono::milliseconds(PAUSE_DELAY));
				seconds = _get_time_span();
				if(next_probe <= seconds){
					Ibroadcast(MSG_REQ_PROBE);
					up_probe = max(up_probe*1.2, PROBE_MAX);
					next_probe = seconds + up_probe;
				}
				process_queue();
			}
			cout << "DEBUG - MPI_Worker0 - Finished to wait for other process" << endl;
		}
		COMM_WORLD.Barrier();
		delete msg;
	}
	return _stop;
}

void DICOD2D::send_result(){
	double cost = compute_cost();
	parentComm->Barrier();
	parentComm->Send(pt, K*L_proc, DOUBLE, ROOT, TAG_MSG_ROOT+world_rank);

	// Gather computed constants
	parentComm->Gather(&cost, UNIT_MSG, DOUBLE, NULL,
					   NULL_SIZE, DOUBLE, ROOT);
	parentComm->Gather(&iter, UNIT_MSG, INT, NULL,
					   NULL_SIZE, DOUBLE, ROOT);
	parentComm->Gather(&runtime, UNIT_MSG, DOUBLE, NULL,
					   NULL_SIZE, DOUBLE, ROOT);
	parentComm->Gather(&t_init, UNIT_MSG, DOUBLE, NULL,
					   NULL_SIZE, DOUBLE, ROOT);

	if (logging){
		double* _log = new double[3*iter];
		list<double>::iterator itz, itt, iti;
		int i = 0;
		for(itz=log_dz.begin(), itt=log_t.begin(), iti=log_i0.begin();
			iti != log_i0.end(); itz++, itt++, iti++, i++){
			_log[3*i] = *iti;
			_log[3*i+1] = *itt;
			_log[3*i+2] = *itz;
		}
		parentComm->Send(_log, 3*iter, DOUBLE, ROOT, TAG_MSG_ROOT+world_rank);
		delete[] _log;
	}
}

double DICOD2D::compute_cost(){
	int s, d, k;
	int h_tau, w_tau, msg_tau, rec_tau, tau;
	int msg_off, rec_off;

	// compute reconstruction size
	int L_rec = h_proc_S*w_proc_S;

	// init workspace and array for convolutions
	Workspace ws;
	init_workspace(ws, LINEAR_FULL, h_proc, w_proc, h_dic, w_dic);
	double *src, *kernel, *rec = new double[dim*L_rec];
	fill(rec, rec+dim*L_rec, 0);

	// init msg holder
	double *val_msg;
	double *msg_right = new double[dim*h_proc_S*(w_dic-1)];
	double *msg_corner = new double[dim*(h_dic-1)*(w_dic-1)];
	double *msg_bottom = new double[dim*(h_dic-1)*w_proc_S];
	double *msg_in_right = new double[dim*h_proc_S*(w_dic-1)];
	double *msg_in_corner = new double[dim*(h_dic-1)*(w_dic-1)];
	double *msg_in_bottom = new double[dim*(h_dic-1)*w_proc_S];
	fill(msg_right, msg_right+dim*h_proc_S*(w_dic-1), 0);
	fill(msg_corner, msg_corner+dim*(h_dic-1)*(w_dic-1), 0);
	fill(msg_bottom, msg_bottom+dim*(h_dic-1)*w_proc_S, 0);

	for(k = 0; k < K; k++)
		for(d = 0; d<dim; d++){
			// compute reconstruction
			src = pt+k*L_proc;
			kernel = D+(k*dim+d)*S;
  			convolve(ws, src, kernel);
  			for(tau=0; tau < L_rec; tau++)
  				rec[d*L_rec + tau] += ws.dst[tau];
  			msg_off = d*h_proc_S*(w_dic-1);
  			for(h_tau=0; h_tau < h_proc_S; h_tau++)
  				for(w_tau=0; w_tau < w_dic-1; w_tau++){
  					msg_tau = h_tau*(w_dic-1)+w_tau;
  					rec_tau = h_tau*w_proc_S+w_tau;
  					msg_right[msg_off+msg_tau] += ws.dst[rec_tau];
  				}
  			msg_off = d*(h_dic-1)*(w_dic-1);
  			for(h_tau=0; h_tau < h_dic-1; h_tau++)
  				for(w_tau=0; w_tau < w_dic-1; w_tau++){
  					msg_tau = h_tau*(w_dic-1)+w_tau;
  					rec_tau = h_tau*w_proc_S+w_tau;
  					msg_corner[msg_off+msg_tau] += ws.dst[rec_tau];
  				}
  			msg_off = d*(h_dic-1)*w_proc_S;
  			for(h_tau=0; h_tau < h_dic-1; h_tau++)
  				for(w_tau=0; w_tau < w_proc_S; w_tau++){
  					msg_tau = h_tau*w_proc_S+w_tau;
  					rec_tau = h_tau*w_proc_S+w_tau;
  					msg_bottom[msg_off+msg_tau] += ws.dst[rec_tau];
  				}
		}
	clear_workspace(ws);
	if(w_rank > 0){
		COMM_WORLD.Isend(msg_right, dim*h_proc_S*(w_dic-1), DOUBLE,
						 world_rank-1, TAG_MSG_COST);
		L_rec = L_proc;
	}
	if(w_rank > 0 && h_rank > 0){
		COMM_WORLD.Isend(msg_corner, dim*(h_dic-1)*(w_dic-1), DOUBLE,
						 world_rank-w_world-1, TAG_MSG_COST);
		L_rec = L_proc;
	}
	if(h_rank > 0){
		COMM_WORLD.Isend(msg_bottom, dim*(h_dic-1)*w_proc_S, DOUBLE,
						 world_rank-w_world, TAG_MSG_COST);
		L_rec = L_proc;
	}
	if(w_rank < w_world-1){
		val_msg = msg_in_right;
		COMM_WORLD.Recv(msg_in_right, dim*h_proc_S*(w_dic-1), DOUBLE,
						world_rank+1, TAG_MSG_COST);
		for(d=0; d< dim; d++){
			rec_off = d*L_rec;
			for(h_tau=0; h_tau < h_proc_S; h_tau++)
  				for(w_tau=0; w_tau < w_dic-1; w_tau++){
  					rec_tau = h_tau*w_proc_S + w_proc + w_tau;
  					rec[rec_off+rec_tau] += *val_msg;
  					val_msg++;
  				}
  		}
	}
	if(h_rank < h_world-1){
		val_msg = msg_in_bottom;
		COMM_WORLD.Recv(msg_in_bottom, dim*(h_dic-1)*w_proc_S, DOUBLE,
						world_rank+w_world, TAG_MSG_COST);
		for(d=0; d < dim; d++){
			rec_off = d*L_rec;
			for(h_tau=0; h_tau < h_dic-1; h_tau++)
  				for(w_tau=0; w_tau < w_proc_S; w_tau++){
  					rec_tau = (h_proc+h_tau)*w_proc_S + w_tau;
  					rec[rec_off+rec_tau] += *val_msg;
  					val_msg++;
  				}
  		}
	}
	if(w_rank < w_world-1 && h_rank < h_world-1){
		val_msg = msg_in_corner;
		COMM_WORLD.Recv(msg_in_corner, dim*(h_dic-1)*(w_dic-1), DOUBLE,
						world_rank+w_world+1, TAG_MSG_COST);
		for(d=0; d < dim; d++){
			msg_off = d*(h_dic-1)*(w_dic-1);
			rec_off = d*L_rec;
			for(h_tau=0; h_tau < h_dic-1; h_tau++)
  				for(w_tau=0; w_tau < w_dic-1; w_tau++){
  					msg_tau = h_tau*(w_dic-1) + w_tau;
  					rec_tau = (h_proc+h_tau)*w_proc_S + w_proc + w_tau;
  					rec[rec_off+rec_tau] += *val_msg;
  					val_msg++;
  				}
  		}
	}
	delete[] msg_in_right;
	delete[] msg_in_corner;
	delete[] msg_in_bottom;

	double a, cost = 0;
	double *its = sig, *itr = rec;
	for(d=0; d < dim; d++){
		its += (h_dic-1)*w_proc_S;
		itr += (h_dic-1)*w_proc_S;
		for(h_tau = 0; h_tau < h_proc; h_tau++){
			its += w_dic-1;
			itr += w_dic-1;
			for(w_tau = 0; w_tau < w_proc; w_tau++){
				a = (*its++ - *itr++);
				cost += a*a;

			}
		}
	}
	if(h_rank == 0){
		its = sig;
		itr = rec;
		for(d=0; d<dim; d++){
			for(h_tau = 0; h_tau < h_dic-1; h_tau++){
				its += w_dic-1;
				itr += w_dic-1;
				for(w_tau = 0; w_tau < w_proc ; w_tau++){
					a = (*its++ - *itr++);
					cost += a*a;
				}
			}
			its += h_proc*w_proc_S;
			itr += h_proc*w_proc_S;
		}
	}
	if(w_rank == 0){
		its = sig;
		itr = rec;
		for(d=0; d<dim; d++){
			its += (h_dic-1)*w_proc_S;
			itr += (h_dic-1)*w_proc_S;
			for(h_tau=0; h_tau < h_proc; h_tau++){
				for(w_tau=0; w_tau<w_dic-1; w_tau++){
					a = (*its++ - *itr++);
					cost += a*a;
				}
			}
		}
	}
	if(h_rank == 0 && w_rank == 0){
		its = sig;
		itr = rec;
		for(d=0; d<dim; d++){
			for(h_tau = 0; h_tau < h_dic-1; h_tau++){
				for(w_tau = 0; w_tau < w_proc; w_tau++){
					a = (*its++ - *itr++);
					cost += a*a;
				}
				its += w_proc;
				itr += w_proc;
			}
			its += h_proc*w_proc_S;
			itr += h_proc*w_proc_S;
		}
	}

	cost /= 2*dim;
	its = pt;
	double z_l1 = 0;
	while(its != pt+K*L_proc)
		z_l1 += fabs(*its++);
	cost += lmbd*z_l1/L;
	COMM_WORLD.Barrier();
	delete[] msg_right;
	delete[] msg_corner;
	delete[] msg_bottom;
	delete[] rec;
	return cost;
}


// send end signals to the neighbors of the processor.
void DICOD2D::_signal_end(){
	if(w_rank < w_world-1)
		_send_msg(world_rank+1, MSG_STOP, 0);				// left neighbor

	if(h_rank < h_world-1){
		if(w_rank < w_world-1)
			_send_msg(world_rank+w_world+1, MSG_STOP, 1);	// lower-left neighbor
		_send_msg(world_rank+w_world, MSG_STOP, 2);			// lower neighbor
		if(w_rank > 0 )
			_send_msg(world_rank+w_world-1, MSG_STOP, 3);	// lower-right neighbor
	}
	if(w_rank > 0)
		_send_msg(world_rank-1, MSG_STOP, 4);				// right neighbor
	if(h_rank > 0){
		if(w_rank > 0 )
			_send_msg(world_rank-w_world-1, MSG_STOP, 5);	// upper-right neighbor
		_send_msg(world_rank-w_world, MSG_STOP, 6);			// upper neighbor
		if(w_rank < w_world-1)
			_send_msg(world_rank-w_world+1, MSG_STOP, 7);	// upper-left neighbor
	}

}

void DICOD2D::end(){
	if((debug || DEBUG) && world_rank == 0)
		cout << "DEBUG - MPI_Workers - flush queue" << endl;

	_signal_end();

	//flush the messages
	Status s;
	int size_msg, src, tag;
	double* msg;
	while(!(end_neigh[0] && end_neigh[1] && end_neigh[2] && end_neigh[3] &&
		    end_neigh[4] && end_neigh[5] && end_neigh[6] && end_neigh[7])){
		COMM_WORLD.Probe(ANY_SOURCE, ANY_TAG, s);
		size_msg = s.Get_count(DOUBLE);
		src = s.Get_source();
		tag = s.Get_tag();
		msg = new double[size_msg];
		COMM_WORLD.Recv(msg, size_msg, DOUBLE, src, tag);
		if(msg[0] == MSG_STOP && msg[1] >= 0)
			end_neigh[(int) msg[1]] = true;
		if(msg[0] == MSG_UP && !go)
			cout << "WARNING - MPI_Worker" << world_rank
				 <<" - Missed wake up" << endl;
		delete[] msg;
	}

	COMM_WORLD.Barrier();
	while(!messages.empty()){
		delete[] messages.front();
		messages.pop_front();
	}
	if((debug || DEBUG) && world_rank == 0)
		cout << "DEBUG - MPI_Workers- Clean operation ok" << endl;
	parentComm->Barrier();
}

// process the message queue
void DICOD2D::process_queue(){
	Status s;
	int size_msg, src, tag;
	double* msg;
	double dz;
	int DD_start, i_try, k0;
	int h_beta_start, w_beta_start, h_ll, w_ll, k, w_tau, h_tau;
	int beta_off, dic_off, l_msg, s_DD, compt = 0;
	int probe_success = 1;
	unordered_map<int, int>::iterator it;
	while(COMM_WORLD.Iprobe(ANY_SOURCE, ANY_TAG, s)){
		compt += 1;
		size_msg = s.Get_count(DOUBLE);
		src = s.Get_source();
		tag = s.Get_tag();
		msg = new double[size_msg];
		COMM_WORLD.Recv(msg, size_msg, DOUBLE, src, tag);
		switch((int) msg[0]){
			case MSG_STOP:
				go = false;
				break;
			case MSG_REQ_PROBE:
				probe_try.push_back((int) msg[1]);
				break;
			case MSG_REP_PROBE:
				l_msg = (int) msg[1];
				for(int i=0; i < l_msg; i++){
					i_try = msg[2+i];
					probe_result[i_try] ++;
					if(probe_result[i_try] == world_size-1 && pause)
						probe_success *= 2;

				}
				break;
			case MSG_HIT_BARRIER:
				n_barrier += 1;
				break;
			case MSG_UP:
				dz = msg[1];
				k0 = (int) msg[2];
				h_beta_start = (h_proc + (int) msg[3])%h_proc;
				w_beta_start = (w_proc + (int) msg[4])%w_proc;
				DD_start = (int) msg[5];
				h_ll = (int) msg[6];
				w_ll = (int) msg[7];
				s_DD = (2*h_dic-1)*(2*w_dic-1);

				// update beta localy
				for(k=0; k < K; k++){
					beta_off = k*L_proc + h_beta_start*w_proc + w_beta_start;
					dic_off = k*K*s_DD + k0*s_DD + DD_start;
					for(h_tau=0; h_tau < h_ll; h_tau++){
						for(w_tau=0; w_tau < w_ll; w_tau++)
							beta[beta_off+w_tau] -= DD[dic_off+w_tau]*dz;
						beta_off += w_proc;
						dic_off += 2*w_dic-1;
					}
				}
				pause = false;
				n_zero = 0;
				probe_success = 0;
				for(it=probe_result.begin(); it != probe_result.end(); it++)
					it->second -= world_size;
				break;
		}
		if(probe_success > 1 && pause){
			Ibroadcast(MSG_STOP);
			go = false;
		}
		delete[] msg;
	}
}
void DICOD2D::Ibroadcast(int msg_t){
	int sz = 2;
	double* msg = new double[sz];
	switch(msg_t){
		case MSG_STOP:
			msg[0] = MSG_STOP;
			msg[1] = -1;
		break;
		case MSG_REQ_PROBE:
			int i_try = probe_result.size();
			probe_result[i_try] = n_barrier;
			msg[0] = MSG_REQ_PROBE;
			msg[1] = i_try;
	}
	for(int i = 1; i < world_size; i ++)
		COMM_WORLD.Send(msg, sz, DOUBLE, i, 3);
	messages.push_back(msg);
}
void DICOD2D::probe_reply(){
	int l_msg = probe_try.size()+2;
	double* msg = new double[l_msg];
	msg[0] = MSG_REP_PROBE;
	msg[1] = l_msg-2;
	list<int>::iterator it;
	int i;
	for(it=probe_try.begin(), i=0;
		it != probe_try.end(); it++, i++)
		msg[i+2] = *it;
	COMM_WORLD.Isend(msg, l_msg, DOUBLE, 0, 4);
	messages.push_back(msg);
	probe_try.clear();
}
void DICOD2D::_send_msg(int dest, int msg_type, int arg){
	int sz = 2;
	double* msg = new double[sz];
	msg[0] = (double) msg_type;
	msg[1] = (double) arg;
	if(dest > -1 && dest < world_size){
		COMM_WORLD.Isend(msg, sz, DOUBLE,
						 dest, TAG_MSG_SERVICE);
		messages.push_back(msg);
	}
	else{
		cout << "ERROR - MPI_Worker" << world_rank
			 << " - tried to send a message to " << dest
			 << "with arg " << arg << endl;
	}
}

double DICOD2D::_get_time_span(){
	time_point t2 = chrono::high_resolution_clock::now();
	d_duration time_span = chrono::duration_cast<d_duration>(t2 - t_start);
  	return time_span.count();
}
