#include "worker.h"
#include "dicod.h"
#include <unistd.h>

Worker::Worker(Intercomm* _parentComm){

	// Get the communicator and the Process API
	parentComm = _parentComm;
	world_size = parentComm->Get_size();	// # processus
	world_rank = parentComm->Get_rank();	// Rank in the processus pool
	if(DEBUG){
		// Get the machine this process is running on
		char* procName = new char[MAX_PROCESSOR_NAME];
		int plen;
		Get_processor_name(procName, plen);
		cout << "Start processor " << world_rank << "/" << world_size
			 << " on " << procName << endl;
		delete[] procName;
	}
}

Worker::~Worker(){
	if(DEBUG && world_rank == 0)
		cout << "DEBUG  - Worker exited nicely" << endl;
}

void Worker::start(){
	parentComm->Barrier();
	control_loop();
}

void Worker::control_loop(){
	int * msg = new int[4];
	char* portname;
	Intercomm comm;
	Intracomm commi;
	DICOD *dcp;
	msg[0] = RUN;
	while(msg[0] > 0){
		Status status;
		bool flag = parentComm->Iprobe(ANY_SOURCE, ANY_TAG, status);
		if(!flag){
			usleep(50);
			continue;
		}
		parentComm->Recv(msg, 4, INT, status.Get_source(),
						 status.Get_tag());
		switch(msg[0]){
		case STOP:
			if(DEBUG && world_rank == 0)
				cout << "Stopping msg" << endl;
		break;
		case RESIZE_SERVER:
			//if(world_rank == 0){
			portname = new char[MAX_PORT_NAME];
			Open_port(INFO_NULL, portname);
			if(world_rank == 0){
				parentComm->Send(portname, MAX_PORT_NAME, CHAR,
								 0, TAG_PORT_MSG);
				cout << "Server portname: " << portname << endl;
			}
			commi = COMM_WORLD.Clone();
			comm = commi.Accept(
				portname, INFO_NULL, 0);
			cout << "Connection done!!!!"<< comm.Get_remote_size()  << endl;
			comm.Disconnect();
			//comm.Free();
		break;
		case RESIZE_CLIENT:
			//if(world_rank == 0){
			portname = new char[MAX_PORT_NAME];
			cout << "Will look up" << endl;
			parentComm->Recv(portname, MAX_PORT_NAME, CHAR,
							 0, TAG_PORT_MSG);
			cout << "Client portname: " << portname << endl;
			//}
			commi = COMM_WORLD.Clone();
			comm = commi.Connect(
				portname, INFO_NULL, 0);
			cout << "Connection done!!!!" << comm.Get_remote_size() << endl;
			comm.Disconnect();
			//comm.Free();
		break;
		case SOLVE_DICOD:
			double dz = 100.;
			dcp = new DICOD(parentComm);
			while(!dcp->stop(dz))
				dz = dcp->step();

			// Send back
			dcp->reduce_pt();
			dcp->end();
			delete dcp;
		break;
		}
	}
}


void delay(int msec){
	
}

