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
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include "convolution_fftw.h"
#include "MPI_operations.h"
using namespace FFTW_Convolution;


#define PAUSE_DELAY 3			// Sleep time for paused process in ms
#define PROBE_UP_START 0.05		// Smallest time between probe for end
#define PROBE_MAX 0.4			// Largest time between probes

#define DEBUG true

//Object handeling the computation
DICOD::DICOD(Intercomm* _parentComm){
	// Init time measurement for initialization
	t_start = chrono::high_resolution_clock::now();

	// setup the stdout
	cout << fixed << setprecision(2);

	parentComm = _parentComm;

	// Initiate arrays
	alpha_k = NULL, DD=NULL, D=NULL;
	sig = NULL, beta = NULL, pt=NULL;
	runtime = 0;
	max_probe = 0;

	// Greetings
	world_size = parentComm->Get_size();	// # processus
	world_rank = parentComm->Get_rank();	// Rank in the processus pool
	if(DEBUG){
		// Get the machine this process is running on
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

// Handle initial communication
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
	S = (int) constants[2];  				// Size of the dictionary
	T = (int) constants[3];					// Size of the signal
	lmbd = constants[4];					// Regularization parameter
	tol = constants[5];						// Convergence tolerance
	t_max = constants[6];					// Maximum time
	i_max = (long int) constants[7];		// maximum # of iterations
	debug = ((int) constants[8] > 0);		// Debug level
	logging = ((int) constants[9] == 1);	// Activate the logging
	n_seg = ((int) constants[10]);			// Use a segmented update
	positive = ((int) constants[11] == 1);	// Use to only activate positive updates
	algo =(int) constants[12];				// Coordinate choice algorihtm
	patience = (int) constants[13];			// Max number of 0 updates in ALGO_RANDOM
	delete[] constants;

	if(world_rank == 0 && (DEBUG || debug))
		cout << "DEBUG - MPI_worker - Start with algorihtm : "
			 << ((ALGO_GS==algo)?"Gauss-Southwell":"Random") << endl;

	L = T-S+1;   // Size of the code
	L_proc = L / world_size + 1;
	proc_off = world_rank*L_proc;
	L_proc = min(proc_off+L_proc, L)-proc_off;
	L_proc_S = L_proc+S-1;

	// Receive the signal to process
	delete[] sig;
	sig = new double[L_proc_S*dim];
	// cout << world_rank << "waiting for " << 100+world_rank 
	// 	 << " size: " << L_proc_S*dim << endl;
	parentComm->Recv(sig, L_proc_S*dim, DOUBLE, 0, 100+world_rank);
	// cout << world_rank << "received" << endl;
	confirm_array(parentComm, sig[0], sig[L_proc_S*dim-1]);

	// Init algo and wait for everyone
	_init_algo();
	parentComm->Barrier();

	chrono::high_resolution_clock::time_point t_end = chrono::high_resolution_clock::now();
	chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t_end - t_start);
	t_init = time_span.count();
	t_start = chrono::high_resolution_clock::now();
}

// Init the algo
void DICOD::_init_algo(){
	int k, t, d, tau;
	double bh;
	double* src, *kernel;

	// FFT tools
	Workspace ws;
	init_workspace(ws, LINEAR_VALID, 1, L_proc_S, 1, S);

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
	log_skip.clear();
	log_time.clear();

	// Init the segment choosing and stoping
	current_seg = 0, n_zero = 0, n_skip = 0;
	seg_dz = 0.;
	seg_size = ceil(L_proc * 1. / n_seg);

	end_neigh = new bool[2];
	end_neigh[0] = (world_rank == 0);
	end_neigh[1] = (world_rank == world_size-1);
}

// On step of the coordinate descent
double DICOD::step(){
	if(pause)
		this_thread::sleep_for(chrono::milliseconds(PAUSE_DELAY));

	process_queue();
	int i, k, t, k_off;
	int k0 = 1, t0 = -1;
	double ak, dz = 0, adz = tol;
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
	if(n_seg > 1){
		seg_start = current_seg*seg_size;
		seg_end = (current_seg+1)*seg_size;
		current_seg += 1;
		if(seg_end >= L_proc)
			current_seg = 0;
		seg_end = min(seg_end, L_proc);
	}

	if(algo == ALGO_GS){
		//Find argmax of |z_i - z'_i|
		for(k = 0; k < K; k++){
			ak = alpha_k[k];
			k_off = k*L_proc;
			for (t=seg_start; t < seg_end; t++){
				i = k_off+t;
				beta_i = -beta[i];
				sign_beta_i = (beta_i >= 0)?1:-1;
				if(positive)
					sign_beta_i = (beta_i >= 0)?1:0;
				beta_i = max(0., fabs(beta_i)-lmbd) * sign_beta_i/ak;
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
		// Uniformly sample a coefficient
		uniform_int_distribution<> dis_t(seg_start, seg_end-1);
		uniform_int_distribution<> dis_k(0, K-1);
		t = dis_t(rng);
		k = dis_k(rng);
		ak = alpha_k[k];
		i = k*L_proc+t;

		// Compute the update
		beta_i = -beta[i];
		sign_beta_i = (beta_i >= 0)?1:-1;
		if(positive)
			sign_beta_i = (beta_i >= 0)?1:0;
		beta_i = max(0., fabs(beta_i)-lmbd)*sign_beta_i/ak;

		// If the update is not null
		if(fabs(beta_i-pt[i]) > tol){
			k0 = k;
			t0 = t;
			dz = pt[i] - beta_i;
			adz = fabs(dz);
		}
	}

	// Increase n_zero
	if(adz <= tol)
		n_zero += 1;
	else
		n_zero = 0;

	// If no coefficient was selected for an update (GREEDY) or
	// the udpate is 0 (RANDOM), directly return and do not update
	// beta or pt.
	if(t0 == -1){
		n_skip ++;
		return _return_dz(0.);
	}

	// Increase iteration count
	iter += 1;

	if(logging){
		chrono::high_resolution_clock::time_point t_end = chrono::high_resolution_clock::now();
		chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t_end - t_start);
		double seconds = time_span.count();

		log_time.push_back(seconds);
		log_dz.push_back(-dz);
		log_skip.push_back(n_skip);
		log_i0.push_back((double) k0*L+proc_off+t0);
	}
	else if (n_seg > 1){
		log_dz.push_back(-dz);
	}

	// Else update the point
	pt[k0*L_proc+t0] -= dz;
	_update_beta(dz, k0, t0);

	// Reset skip counter
	n_skip = 0;

	return _return_dz(dz);
}

// Handle the computation to avoid ending computation before convergence in
// Random algorithm and in segmented iterations
double DICOD::_return_dz(double dz){

	if(n_seg > 1){
		// For segmented algorithm, return the max of dz
		// over the n_seg last updates
		int k;
		list<double>::reverse_iterator rit;

		for(rit = log_dz.rbegin(), k = 0;
			rit != log_dz.rend() && k < 2 * n_seg - n_zero;
			rit++, k++)
			if(fabs(*rit) > fabs(dz))
				dz = *rit;
	}
	if(algo == ALGO_RANDOM)
		// For the RANDOM algorithm, wait until we get a number of
		// 0 updates superior to patience
		if(patience > n_zero)
			return 2 * tol;
		else
			return _check_convergence();

	return dz;
}

// Update the beta variable after a change of coordinate (k0, t0) of dz
void DICOD::_update_beta(double dz, int k0, int t0){
	if(dz == 0)
		return;
	//Loop variables
	int k, tau;
	//Offset variables
	int beta_off, DD_off, s_DD= 2*S-1;
	int DD_start, cod_start, ll;
	//Hold previous beta for the current indice
	int i0 = k0*L_proc+t0;
	double p_beta_i0 = beta[i0];

	// Compute offset
	DD_start = max(0, S-t0-1);
	cod_start = max(0, t0-S+1);
	ll = min(L_proc, t0+S) - cod_start;

	// Update local beta coefficients
	beta_off = cod_start;
	DD_off = k0*s_DD + DD_start;
	for(k=0; k < K; k++){
		for(tau=0; tau < ll; tau++)
			beta[beta_off+tau] -= DD[DD_off+tau]*dz;
		beta_off += L_proc;
		DD_off += K*s_DD;
	}
	beta[i0] = p_beta_i0;
	if (DD_start > 0 && world_rank > 0)
		send_update_msg(world_rank-1, dz, k0, -DD_start, 0, DD_start);
	else if (t0 > L_proc-S && world_rank < world_size-1)
		send_update_msg(world_rank+1, dz, k0, 0, ll, s_DD-ll);
}
void DICOD::send_update_msg(int dest, double dz, int k0,
							int cod_start, int DD_start, int ll)
{
	double* msg = new double[HEADER];
	msg[0] = (double) UP;
	msg[1] = dz;
	msg[2] = (double) k0;
	msg[3] = (double) cod_start;
	msg[4] = (double) DD_start;
	msg[5] = (double) ll;
	COMM_WORLD.Isend(msg, HEADER, DOUBLE,
					 dest, TAG_UP);
	messages.push_back(msg);
}

double DICOD::_check_convergence(){

	int k, t, i = 0;
	int sign_b;
	double b, ak, dz, adz = 0;
	//Find argmax of |z_i - z'_i|
	for(k = 0; k < K; k++){
		ak = alpha_k[k];
		for (t=0; t < L_proc; t++){
			b = -beta[i];
			sign_b = (b >= 0)?1:-1;
			if(positive)
				sign_b = (b >= 0)?1:0;
			dz = fabs(max(0., fabs(b) - lmbd)*sign_b/ak - pt[i]);
			adz = max(adz, dz);
			if (adz > tol)
				break;
			i ++;
		}
	}
	if (adz > tol)
		n_zero = 0;
	return adz;
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
			cout << "\nINFO - MPI_worker - Reach optimal solution in "
				 << seconds << endl;
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
	if(world_rank == 0 && (debug || DEBUG) &&  (iter % 1000 == 0 || pause)){
		double progress = max(iter * 100.0 / i_max, seconds * 100.0 / t_max);
		cout << "\rDEBUG - MPI_worker - Progress " << setw(2)
			 << progress << "%   (probe: " << max_probe << ")" << flush;

	}
	return _stop;
}

void DICOD::reduce_pt(){
	long int i;
	double cost = compute_cost();
	parentComm->Barrier();
	parentComm->Send(pt, L_proc*K, DOUBLE, 0, 200+world_rank);
	parentComm->Gather(&cost, 1, DOUBLE, NULL, 0, DOUBLE, 0);
	parentComm->Gather(&iter, 1, INT, NULL, 0, INT, 0);
	parentComm->Gather(&runtime, 1, DOUBLE, NULL, 0, DOUBLE, 0);
	parentComm->Gather(&t_init, 1, DOUBLE, NULL, 0, DOUBLE, 0);

	if (logging){
		parentComm->Barrier();
		double* _log = new double[4*iter];
		list<double>::iterator it_dz, it_time, it_i0, it_skip;
		for(i=0, it_dz=log_dz.begin(), it_time=log_time.begin(),
			it_i0=log_i0.begin(), it_skip=log_skip.begin();
			it_i0 != log_i0.end();
			i++, it_dz++, it_i0++, it_time++, it_skip++){
			_log[4*i] = *it_i0;
			_log[4*i+1] = *it_time;
			_log[4*i+2] = *it_dz;
			_log[4*i+3] = *it_skip;
		}
		parentComm->Send(_log, 4*iter, DOUBLE, 0, 300+world_rank);
		delete[] _log;
	}
}

double DICOD::compute_cost(){
	Workspace ws;
	int s, d, k, tau;
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
	if(world_rank > 0){

		COMM_WORLD.Isend(msg, dim*(S-1), DOUBLE, world_rank-1, TAG_MSG_COST);
	}
	if(world_rank < world_size-1){
		COMM_WORLD.Recv(msg_in, dim*(S-1), DOUBLE, world_rank+1, TAG_MSG_COST);
		for(d=0; d< dim; d++)
			for(s=0; s < S-1; s++)
				rec[d*L_proc_S+s] += msg_in[d*(S-1)+s];
	}
	delete[] msg_in;
	double a, cost, Er = 0;
	double *its = sig, *itr = rec;
	int L_off = S-1, L_ll = L_proc;
	if(world_rank == 0){
		L_off = 0;
		L_ll = L_proc_S;
	}
	for(d=0; d < dim; d++){
		its += L_off;
		itr += L_off;
		for(tau = 0; tau < L_ll; tau++){
			a = (*its++ - *itr++);
			Er += a*a;
		}
	}

	Er /= 2*dim;
	cout.precision(10);
	double z_l1 = 0;
	its = pt;
	while(its != pt+K*L_proc)
		z_l1 += fabs(*its++);
	cost = Er + lmbd*z_l1/L;
	COMM_WORLD.Barrier();
	delete[] msg;
	delete[] rec;
	return cost;
}

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
	int ll, k, tau, i_try, l_msg;
	int DD_off, beta_off, k0, DD_start, cod_start, s_DD;
	int compt = 0, probe_val;
	double dz;
	while(COMM_WORLD.Iprobe(ANY_SOURCE, ANY_TAG, s) && (compt < 10000)){
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
			case REQ_PROBE:
				probe_try.push_back((int) msg[1]);
				break;
			case REP_PROBE:
				l_msg = (int) msg[1];
				for(int i=0; i < l_msg; i++){
					i_try = msg[2+i];
					probe_val = ++probe_result[i_try];
					if(probe_val == world_size && pause){
						Ibroadcast(STOP);
						go = false;
					}
					max_probe = max(max_probe, probe_val);
				}
				break;
			case UP:
				dz = msg[1];
				k0 = (int) msg[2];
				cod_start = (L_proc + (int) msg[3])%L_proc;
				DD_start = (int) msg[4];
				ll = (int) msg[5];
				s_DD = 2*S-1;

				for(k=0; k<K; k++){
					beta_off = k*L_proc + cod_start;
					DD_off = k*K*s_DD + k0*s_DD + DD_start;
					for(tau=0; tau< ll; tau ++)
						beta[beta_off+tau] -= dz*DD[DD_off+tau];
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
			probe_result[i_try] = 1;
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
