#ifndef _PLEXREAD_H_
#define _PLEXREAD_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>

#include "plexfile.h"

//Spikes are not strictly in monotonic order, so a binary search may miss a few at the edges
//This defines how far on each side to search for spikes within the range
#define SPIKE_SEARCH 200

typedef struct ContInfo {
    unsigned long len;
    unsigned long nchans;
    double t_start;
    double start;
    double stop;
    int freq;

    int* chans;
    int* cskip;
    
    ChanType type;
    PlexFile* plxfile;
    long long _fedge[2];
    unsigned long _strunc[2];
    unsigned long _start;
} ContInfo;

typedef struct Spike {
    double ts;
    int chan;
    int unit;
} Spike;

typedef struct SpikeInfo {
    int num;
    short wflen;
    double start;
    double stop;

    ChanType type;
    PlexFile* plxfile;
    unsigned long _fedge[2];
} SpikeInfo;

extern ContInfo* plx_get_continuous(PlexFile* plxfile, ChanType type,
    double start, double stop, int* chans, int nchans);
extern void plx_read_continuous(ContInfo* info, double* data);
extern void free_continfo(ContInfo* info);

extern SpikeInfo* plx_get_discrete(PlexFile* plxfile, ChanType type, 
    double start, double stop);
extern void plx_read_discrete(SpikeInfo* info, Spike* data);
extern void plx_read_waveforms(SpikeInfo* info, double* data);
extern void free_spikeinfo(SpikeInfo* info);

long long _binary_search(FrameSet* frameset, TSTYPE ts);


#endif