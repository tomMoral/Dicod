
#ifndef CONSTANTS_H
#define CONSTANTS_H

//Communication messages
#define MSG_STOP		0
#define MSG_UP			1
#define MSG_REQ_PROBE	2
#define MSG_REP_PROBE	3
#define MSG_HIT_BARRIER	4

// Coordinate choice algorithms
#define ALGO_GS		0
#define ALGO_RANDOM	1

// Worker states
#define WORKER_STATE_RUNNING	0
#define WORKER_STATE_PAUSE		1
#define WORKER_STATE_STOP		2

// Communication message tags
#define TAG_MSG_ROOT 	4242
#define TAG_MSG_UP 		2742
#define TAG_MSG_SERVICE 2727
#define TAG_MSG_COST	3615
#define TAG_MSG_AB		4227

// Define some constants
#define ROOT 		0
#define UNIT_MSG	1
#define NULL_SIZE 	0
#define HEADER_2D 	15

#endif
