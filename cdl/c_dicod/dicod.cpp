//
// MPI implementation of Dicod, a distributed algorithm to compute
// convolutional sparse coding, based on coordinate descent.
// See [Moreau2015] for details
//
// Author: Thomas Moreau  (thomas.moreau@cmla.ens-cachan.fr)
// Date: May 2015
//
#include "dicod.h"

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "convolution_fftw.h"
#include "MPI_operations.h"
using namespace FFTW_Convolution;


#define PAUSE_DELAY 3			// Sleep time for paused process in ms
#define PROBE_UP_START 0.05		// Smallest time between probe for end
#define PROBE_MAX 0.4			// Largest time between probes

#define DEBUG false

//Object handeling the computation
DICOD::DICOD(Intercomm* _parentComm){
	// Init time measurement for initialization
	t_start = chrono::high_resolution_clock::now();

	parentComm = _parentComm;

	// Initiate arrays
	alpha_k = NULL, DD=NULL, D=NULL;
	sig = NULL, beta = NULL, pt=NULL;
	runtime = 0;

	// Greetings
	world_size = parentComm->Get_size();	// # processus
	world_rank = parentComm->Get_rank();	// Rank in the processus pool
	if(DEBUG){
		// Get the machine this process is running on
		char* procName = new char[MAX_PROCESSOR_NAME];
		int plen;
		Get_processor_name(procName, plen);
		cout << "Start processor " << world_rank << "/" << world_size
			 << " on " << procName << endl
			 << "    and with COMM_WORLD " << COMM_WORLD.Get_rank() << "/" 
			 << COMM_WORLD.Get_size() << endl;
		delete[] procName;
	}
	int seed =  chrono::duration_cast<chrono::milliseconds>(
                   t_start.time_since_epoch()).count();
	rng.seed(world_rank*seed);

	this->receive_task();
}
// Destructor, delete all arrays
DICOD::~DICOD(){
	delete[] D;
	delete[] DD;
	delete[] alpha_k;
	delete[] sig;
	delete[] pt;
	delete[] beta;
	delete[] end_neigh;
}
// Init the algo
void DICOD::_init_algo(){
	int k, t, d, tau;
	double bh;

	// FFT tools
	Workspace ws;
	init_workspace(ws, LINEAR_VALID, 1, L_proc_S, 1, S);

 	// The main loop

	//Initialize beta and pt
	delete[] beta;
	delete[] pt;
	beta = new double[K*L_proc];
	pt = new double[K*L_proc];
	fill(beta, beta+K*L_proc, 0);
	fill(pt, pt+K*L_proc, 0);
	double* src, *kernel = new double[S];
	for(k = 0; k < K; k++){
		for(d=0; d < dim; d++){
			src = &sig[d*L_proc_S];
			//Revert dic
			for(tau=0; tau < S; tau++)
				kernel[tau] = D[k*dim*S+(d+1)*S-tau-1];
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
	up_probe = PROBE_UP_START;
	log_dz.clear();
	log_i0.clear();
	log_t.clear();

	// Init the segment choosing and stoping
	current_seg = 0;
	seg_dz = 0.;
	n_seg = use_seg;
	seg_size = ceil(L_proc / use_seg);
	n_zero = 0;

	end_neigh = new bool[2];
	end_neigh[0] = (world_rank == 0);
	end_neigh[1] = (world_rank == world_size-1);

}
double DICOD::step(){
	if(pause)
		this_thread::sleep_for(chrono::milliseconds(PAUSE_DELAY));
	process_queue();
	int i, k, t;
	int k0, t0 = -1;
	double ak, dz, adz = tol;
	double beta_i, sign_beta_i;
	if(pause && probe_try.size() > 0){
		probe_reply();
		return tol;
	}
	probe_try.clear();

	// Compute the current segment if we use the 
	// segmented version of the algorithm
	int seg_start = 0;
	int seg_end = L_proc;
	if(use_seg > 1){
		seg_start = current_seg*seg_size;
		seg_end = (current_seg+1)*seg_size;
		current_seg += 1;
		if(seg_end > L_proc){
			current_seg = 0;

		}
		seg_end = min(seg_end, L_proc);
	}

	if(algo == ALGO_GS){
		//Find argmax of |z_i - z'_i|
		for(k = 0; k < K; k++){
			ak = alpha_k[k];
			for (t=seg_start; t < seg_end; t++){
				i = k*L_proc+t;
				beta_i = -beta[i];
				sign_beta_i = (beta_i >= 0)?1:-1;
				if(positive)
					sign_beta_i = (beta_i >= 0)?1:0;
				beta_i = max(0., fabs(beta_i)-lmbd)*sign_beta_i/ak;
				if(adz < fabs(beta_i-pt[i])){
					k0 = k;
					t0 = t;
					dz = pt[i]-beta_i;
					adz = fabs(dz);
				}
			}
		}
	}
	else if(algo == ALGO_RANDOM){
		uniform_int_distribution<> dis_t(seg_start, seg_end-1);
		uniform_int_distribution<> dis_k(0, K-1);
		t = dis_t(rng);
		k  = dis_k(rng);
		ak = alpha_k[k];
		i = k*L_proc+t;
		beta_i = -beta[i];
		sign_beta_i = (beta_i >= 0)?1:-1;
		if(positive)
			sign_beta_i = (beta_i >= 0)?1:0;
		beta_i = max(0., fabs(beta_i)-lmbd)*sign_beta_i/ak;
		if(adz < fabs(beta_i-pt[i])){
			k0 = k;
			t0 = t;
			dz = pt[i]-beta_i;
			adz = fabs(dz);
		}
	}
	// If there is no update
	// it will go to pause in end
	if(t0 == -1){
		n_zero += 1;
		return _return_dz(0.);;
	}
	n_zero = 0;
	cout << "Update t: " << t0 << endl;

	// Else update the point
	pt[k0*L_proc+t0] -= dz;
	if(logging){
		//int off = world_rank*L_proc;
		chrono::high_resolution_clock::time_point t_end = chrono::high_resolution_clock::now();
		chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t_end - t_start);
	  	double seconds = time_span.count();
	
		log_t.push_back(seconds);
		log_dz.push_back(-dz);
		log_i0.push_back((double) k0*L+off+t0);
	}
	_update_beta(dz, k0, t0);
	iter += 1;

	return _return_dz(dz);
}

double DICOD::_return_dz(double dz){

	if(use_seg > 1){
		// For semgented algorithm, return the max of dz
		// over the n_seg last updates
		int k;
		list<double>::reverse_iterator rit;

		for(rit = log_dz.rbegin(), k = 0;
			rit != log_dz.rend() && k != 2*n_seg-n_zero;
			rit++, k++)
			if(fabs(*rit) > fabs(dz))
				dz = *rit;
	}
	if(algo == ALGO_RANDOM)
		// For the RANDOM algorithm, wait until we get a number of 
		// 0 updates superior to patience
		if(patience > n_zero)
			dz = 2*tol;
	return dz;
}

bool DICOD::stop(double dz){
	chrono::high_resolution_clock::time_point t_end = chrono::high_resolution_clock::now();
	chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t_end - t_start);
  	double seconds = time_span.count();
	bool _stop = false;
	_stop |= (iter >= i_max);
	_stop |= (seconds >= t_max);
	if(!go){
		COMM_WORLD.Barrier();
		if(world_rank == 0 && (debug || DEBUG))
			cout << "INFO - MPI_worker - Reach optimal solution in " << seconds << endl;
		return true;
	}
	if((debug || DEBUG) && seconds >= t_max && world_rank == 0)
		cout << "DEBUG - MPI_worker - Reach timeout" << endl;
	if((debug || DEBUG) && iter >= i_max && world_rank == 0)
		cout << "DEBUG - MPI_worker - Reach max iteration" << endl;
	if(fabs(dz) <= tol){
		// If just enter pause, probe other for paused
		if(world_rank == 0){
			if(world_size == 1){
				go = false;
				pause = true;
				runtime = seconds;
				return true;
			}
			if(!pause){
				up_probe = PROBE_UP_START;
				next_probe = seconds;
				pause = true;
				runtime = seconds;
			}
			if(next_probe <= seconds){
				Ibroadcast(REQ_PROBE);
				up_probe = max(up_probe*1.2, PROBE_MAX);
				next_probe = seconds + up_probe;
			}
		}
		else if(!pause){
			pause = true;
			runtime = seconds;
		}
	}
	if(_stop){
		if(runtime == 0)
			runtime = seconds;
		COMM_WORLD.Barrier();
	}
	return _stop;
}
void DICOD::_update_beta(double dz, int k0, int t0){
	//Loop variables
	int k, tau;
	int kk, dk, s_DD= 2*S-1;
	//Offset variables
	int off, start, ll;
	//Hold previous beta for the current indice
	int i0 = k0*L_proc+t0;
	double p_beta_i0 = beta[i0];
	off = max(0, S-t0-1);
	start = max(0, t0-S+1);
	ll = min(L_proc,t0+S)-start;
	for(k=0; k < K; k++){
		kk = k*L_proc;
		dk = k*K*s_DD+k0*s_DD;
		for(tau=0; tau < ll; tau++){
			beta[kk+start+tau] -= DD[dk+off+tau]*dz;
		}
	}
	beta[i0] = p_beta_i0;
	if((off > 0 && world_rank != 0) ||
	   		(t0+S > L_proc && world_rank != world_size-1)){
		send_update(dz, ll, off, k0);
	}
}
void DICOD::reduce_pt(){
	double cost = compute_cost();
	parentComm->Barrier();
	parentComm->Send(pt, L_proc*K, DOUBLE, 0, 200+world_rank);
	parentComm->Gather(&cost, 1, DOUBLE, NULL, 0, DOUBLE, 0);
	parentComm->Gather(&iter, 1, INT, NULL, 0, DOUBLE, 0);
	parentComm->Gather(&runtime, 1, DOUBLE, NULL, 0, DOUBLE, 0);
	parentComm->Gather(&t_init, 1, DOUBLE, NULL, 0, DOUBLE, 0);

	if (logging){
		parentComm->Barrier();
		double* _log = new double[3*iter];
		list<double>::iterator itz, itt, iti;
		int i = 0;
		for(itz=log_dz.begin(), itt=log_t.begin(), iti=log_i0.begin();
			iti != log_i0.end(); itz++, itt++, iti++, i++){
			_log[3*i] = *iti;
			_log[3*i+1] = *itt;
			_log[3*i+2] = *itz;
		}
		parentComm->Send(_log, 3*iter, DOUBLE, 0, 300+world_rank);
		delete[] _log;
	}
}
double DICOD::compute_cost(){
	Workspace ws;
	int s, d, k;
	int L_rec = L_proc_S;
	init_workspace(ws, LINEAR_FULL, 1, L_proc, 1, S);
	double *msg = new double[(S-1)*dim], *msg_in = new double[(S-1)*dim];
	double *src, *kernel, *rec = new double[dim*L_proc_S];
	fill(rec, rec+dim*L_proc_S, 0);
	fill(msg, msg+dim*(S-1), 0);
	for(k = 0; k < K; k++)
		for(d = 0; d<dim; d++){
			src = &pt[k*L_proc];
			kernel = &D[(k*dim+d)*S];
  			convolve(ws, src, kernel);
  			for(s=0; s < L_proc_S; s++)
  			 	rec[d*L_proc_S+s] += ws.dst[s];
  			for(s=0; s < S-1; s++)
  				msg[d*(S-1)+s] += ws.dst[s];
		}
	if(world_rank != world_size-1){

		COMM_WORLD.Isend(msg, dim*(S-1), DOUBLE, world_rank+1, 27);
		L_rec = L_proc;
	}
	if(world_rank != 0){
		COMM_WORLD.Recv(msg_in, dim*(S-1), DOUBLE, world_rank-1, 27);
		for(d=0; d< dim; d++)
			for(s=0; s < S-1; s++)
				rec[d*L_proc+s] += msg_in[d*(S-1)+s];
	}
	delete[] msg_in;
	double a, cost = 0;
	double *its = sig, *itr = rec;
	while(its != sig+dim*L_proc_S){
		a = (*its++ - *itr++);
		cost += a*a;
	}
	a /= 2*dim;
	its = pt;
	while(its != pt+K*L_proc)
		cost += lmbd*fabs(*its++);
	COMM_WORLD.Barrier();
	delete[] msg;
	delete[] rec;
	return cost;
}

// bool DICOD::check_dest(int dest){
// 	if(dest > -1 && dest < world_size){
// 		COMM_WORLD.Isend(msg, HEADER, DOUBLE,
// 						 dest, 34+(1-2*src));
// 		messages.push_back(msg);
// 	}
// 	else{
// 		cout << "ERROR - MPI_worker - tried to send a message to" << dest << endl;
// 	}
// }

void DICOD::end(){
	if((debug || DEBUG) && world_rank == 0)
		cout << "DEBUG - MPI_worker - flush queue" << endl;
	if(world_rank != 0)
		send_msg(STOP, 1, false);
	if(world_rank != world_size-1)
		send_msg(STOP, 0, true);

	//flush the messages
	Status s;
	int size_msg, src, tag;
	double* msg;
	while(!(end_neigh[0] && end_neigh[1])){
		COMM_WORLD.Probe(ANY_SOURCE, ANY_TAG, s);
		size_msg = s.Get_count(DOUBLE);
		src = s.Get_source();
		tag = s.Get_tag();
		msg = new double[size_msg];
		COMM_WORLD.Recv(msg, size_msg, DOUBLE, src, tag);
		if(msg[0] == STOP && msg[1] >= 0)
			end_neigh[(int) msg[1]] = true;
		if((debug || DEBUG) && msg[0] == UP && !go)
			cout << "WARNING - MPI_worker" << world_rank
				 <<" - Missed wake up" << endl;
		delete[] msg;
	}

	COMM_WORLD.Barrier();
	while(!messages.empty()){
		delete[] messages.front();
		messages.pop_front();
	}
	if((debug || DEBUG) && world_rank == 0)
		cout << "DEBUG - MPI_worker - Clean operation ok" << endl;
	parentComm->Barrier();
}
// Process the message queue
void DICOD::process_queue(){
	Status s;
	int size_msg, src, tag;
	double* msg;
	int from, off, ll, k, tau, i_try, l_msg;
	int compt = 0;
	while(COMM_WORLD.Iprobe(ANY_SOURCE, ANY_TAG, s)&&(compt < 10000)){
		compt += 1;
		size_msg = s.Get_count(DOUBLE);
		src = s.Get_source();
		tag = s.Get_tag();
		msg = new double[size_msg];
		COMM_WORLD.Recv(msg, size_msg, DOUBLE, src, tag);
		switch((int) msg[0]){
			case STOP:
				go = false;
				break;
			case PAUSE:
				from = (int) msg[1];
				break;
			case REQ_PROBE:
				probe_try.push_back((int) msg[1]);
							break;
			case REP_PROBE:
				l_msg = (int) msg[1];
				for(int i=0; i < l_msg; i++){
					i_try = msg[2+i];
					probe_result[i_try] ++;
					if(probe_result[i_try] == world_size-1 && pause){
						Ibroadcast(STOP);
						go = false;
					}
				}
				break;
			case UP:
				int dk, k0, start_DD, s_DD = 2*S-1;
				double dz;
				from = (int) msg[1];
				off = (L_proc + (int) msg[2])%L_proc;
				ll = (int) msg[3];
				k0 = (int) msg[4];
				start_DD = (int) msg[5];
				dz = msg[6];
				for(k=0; k<K; k++){
					dk = k*K*s_DD + k0*s_DD; 
					for(tau=0; tau< ll; tau ++){
						beta[k*L_proc+off+tau] -= dz*DD[dk+start_DD+tau];
					}
				}
				pause = false;
				runtime = 0;
				n_zero = 0;
				break;
		}
		delete[] msg;
	}
}
void DICOD::Ibroadcast(int msg_t){
	int sz = 2;
	double* msg = new double[sz];
	switch(msg_t){
		case STOP:
			msg[0] = STOP;
			msg[1] = -1;
		break;
		case REQ_PROBE:
			int i_try = probe_result.size();
			probe_result[i_try] = 0;
			msg[0] = REQ_PROBE;
			msg[1] = i_try;
	}
	for(int i = 1; i < world_size; i ++)
		COMM_WORLD.Send(msg, sz, DOUBLE, i, 3);
	messages.push_back(msg);
}
void DICOD::probe_reply(){
	int l_msg = probe_try.size()+2;
	double* msg = new double[l_msg];
	msg[0] = REP_PROBE;
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
void DICOD::send_update(double dz, int ll, int off, int k0){

	int src, start, l_msg, start_DD;
	int k, tau, dk, s_DD = 2*S-1;
	if(off > 0){
		src = 1;
		start = -off;
		start_DD = 0;
		l_msg = off;
	}
	else{
		src = 0;
		start = 0;
		start_DD = ll;
		l_msg = s_DD-ll;
	}
	double* msg = new double[HEADER];
	msg[0] = (double) UP;
	msg[1] = (double) src;
	msg[2] = (double) start;
	msg[3] = (double) l_msg;
	msg[4] = (double) k0;
	msg[5] = (double) start_DD;
	msg[6] = dz;
	int dest = world_rank+(1-2*src);
	if(dest > -1 && dest < world_size){
		COMM_WORLD.Isend(msg, HEADER, DOUBLE,
						 dest, 34+(1-2*src));
		messages.push_back(msg);
	}
	else{
		cout << "ERROR - MPI_worker" << world_rank 
			 << " - tried to send a message to" << dest << endl;
	}
}
void DICOD::send_msg(int msg_type, int arg, bool up){
	int sz = 2;
	double* msg = new double[sz];
	msg[0] = (double) msg_type;
	msg[1] = (double) arg;
	int dest = world_rank+(2*up-1);
	if(dest > -1 && dest < world_size){
		COMM_WORLD.Isend(msg, sz, DOUBLE,
						 dest, 34+(2*up-1));
		messages.push_back(msg);
	}
	else{
		cout << "ERROR - MPI_worker" << world_rank 
			 << " - tried to send a message to" << dest << endl;
	}
}
void DICOD::receive_task(){

	// Update dictionary constants
	delete[] alpha_k;
	delete[] DD;
	delete[] D;
	alpha_k = receive_bcast(parentComm);
	DD = receive_bcast(parentComm);
	D = receive_bcast(parentComm);

	//Receives some constant of the algorithm
	double* constants = receive_bcast(parentComm);
	dim = (int) constants[0];				// Dimension of the signal
	K = (int) constants[1];					// Number of dictionary elements
	S = (int) constants[2];  				// Size of the dicitonary
	T = (int) constants[3];					// Size of the signal
	lmbd = constants[4];					// Regularisation parameter
	tol = constants[5];						// Convergence tolerance
	t_max = constants[6];					// Maximum time
	i_max = (int) constants[7];				// # iterations maximum
	debug = ((int) constants[8] > 0);		// Debug level
	logging = ((int) constants[9] == 1);	// Activate the logging
	use_seg = ((int) constants[10]);		// Use a semgneted update
	positive = ((int) constants[11] == 1);	// Use to only activate positive updates
	algo =(int) constants[12];				// Coordinate choice algorihtm
	patience = (int) constants[13];			// Max number of 0 updates in ALGO_RANDOM
	delete[] constants;

	if(world_rank == 0 && (DEBUG || debug))
		cout << "DEBUG - MPI_worker - Start with algoirhtm : " 
			 << ((ALGO_GS==algo)?"Gauss-Southwell":"Random") << endl;

	L = T-S+1;   // Size of the code
	L_proc = L / world_size + 1;
	off = world_rank*L_proc;
	L_proc = min(off+L_proc, L)-off;
	L_proc_S = L_proc+S-1;

	// Receive the signal to process
	delete[] sig;
	sig = new double[L_proc_S*dim];
	parentComm->Recv(sig, L_proc_S*dim, DOUBLE, 0, 100+world_rank);
	confirm_array(parentComm, sig[0], sig[L_proc_S*dim-1]);

	// Init algo and wait for everyone
	_init_algo();
	parentComm->Barrier();

	chrono::high_resolution_clock::time_point t_end = chrono::high_resolution_clock::now();
	chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t_end - t_start);
  	t_init = time_span.count();
	t_start = chrono::high_resolution_clock::now();
}