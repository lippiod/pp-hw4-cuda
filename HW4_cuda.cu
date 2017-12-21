#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda.h>
#include <unistd.h>
#include <sys/time.h>

#define CEIL(a, b) ( ((a) + (b) - 1) / (b) )
#define MIN(a, b) ( (a) < (b) ? (a) : (b) )
#define CAL_TIME ( 1e-6 * (temp_time.tv_usec - start_time.tv_usec) + (temp_time.tv_sec - start_time.tv_sec) )
#define C2I(i) ( fstrp[i] - '0')

#define INF 1000000000

void input(char *inFileName);
void output(char *outFileName);
void block_FW();
void print();
void split_strings(int m, char *sstart);

int n, n_bytes; // Number of vertices, edges
int Rounds, b_rounds, b_rounds_bytes, dist_size, dist_size_bytes;
unsigned int *Dist_h;
char buf[INF];
char *fstr;

struct timeval start_time, temp_time;

unsigned int *Dist_d;
size_t pitch_d, pitch_int;
__constant__ unsigned int *Dist;
__constant__ size_t pitch;

//const int tpb = 1024;
const int block_size = 32;

cudaStream_t stream[16];

__global__ void cal_phase1(int pivot)
{
    __shared__ unsigned int block_dist[block_size][block_size];

    int tidx = threadIdx.x, tidy = threadIdx.y;
    int block_index = pivot + pitch * tidy + tidx;

    block_dist[tidy][tidx] = Dist[block_index];
    __syncthreads();

    for(int k=0; k<block_size; k++) {
        unsigned int new_dist = block_dist[tidy][k] + block_dist[k][tidx];
        if (block_dist[tidy][tidx] > new_dist)
            block_dist[tidy][tidx] = new_dist;
        __syncthreads();
    }

    Dist[block_index] = block_dist[tidy][tidx];
}

__global__ void cal_phase2_row(int r)
{
    __shared__ unsigned int block_dist[block_size][block_size];
    __shared__ unsigned int pivot_dist[block_size][block_size];

    int tidx = threadIdx.x, tidy = threadIdx.y;
    int pivot = block_size * r;
    int py = pitch * (pivot + tidy);
    int px = pivot + tidx;
    int block_index, pivot_index;

    if(blockIdx.x==r) // pivot block
        return;

    block_index = py + block_size * blockIdx.x + tidx;
    pivot_index = py + px;

    block_dist[tidy][tidx] = Dist[block_index];
    pivot_dist[tidy][tidx] = Dist[pivot_index];
    __syncthreads();

    for(int k=0; k<block_size; k++) {
        unsigned int new_dist = pivot_dist[tidy][k] + block_dist[k][tidx];

        if (block_dist[tidy][tidx] > new_dist)
            block_dist[tidy][tidx] = new_dist;
        __syncthreads();
    }

    Dist[block_index] = block_dist[tidy][tidx];
}

__global__ void cal_phase2_col(int r)
{
    __shared__ unsigned int block_dist[block_size][block_size];
    __shared__ unsigned int pivot_dist[block_size][block_size];

    int tidx = threadIdx.x, tidy = threadIdx.y;
    int pivot = block_size * r;
    int py = pitch * (pivot + tidy);
    int px = pivot + tidx;
    int block_index, pivot_index;

    if(blockIdx.x==r) // pivot block
        return;

    block_index = pitch * (block_size * blockIdx.x + tidy) + px;
    pivot_index = py + px;

    block_dist[tidy][tidx] = Dist[block_index];
    pivot_dist[tidy][tidx] = Dist[pivot_index];
    __syncthreads();

    for(int k=0; k<block_size; k++) {
        unsigned int new_dist = block_dist[tidy][k] + pivot_dist[k][tidx];

        if (block_dist[tidy][tidx] > new_dist)
            block_dist[tidy][tidx] = new_dist;
        __syncthreads();
    }

    Dist[block_index] = block_dist[tidy][tidx];
}

__global__ void cal_phase3(int r)
{
    __shared__ unsigned int pvRow_dist[block_size][block_size];
    __shared__ unsigned int pvCol_dist[block_size][block_size];

    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bx = block_size * blockIdx.x, by = block_size * blockIdx.y;
    int pv = block_size * r;
    int block_index, pvRow_index, pvCol_index;
    unsigned int block_dist;

    if(blockIdx.y==r || blockIdx.x==r) // pivots
        return;

    pvRow_index = pitch * (pv + tidy) + bx + tidx;
    pvRow_dist[tidy][tidx] = Dist[pvRow_index];

    pvCol_index = pitch * (by + tidy) + pv + tidx;
    pvCol_dist[tidy][tidx] = Dist[pvCol_index];
    __syncthreads();

    block_index = pitch * (by + tidy) + bx + tidx;
    block_dist = Dist[block_index];

    for(int k=0; k<block_size; k++) {
        unsigned int new_dist = pvCol_dist[tidy][k] + pvRow_dist[k][tidx];
        if (block_dist > new_dist)
            block_dist = new_dist;
    }

    Dist[block_index] = block_dist;
}

__global__ void set_inf()
{
    int dist_index = pitch * (block_size * blockIdx.y + threadIdx.y) + block_size * blockIdx.x + threadIdx.x;
    unsigned int value = Dist[ dist_index ];

    if (value>=INF)
        value = INF;

    Dist[ dist_index ] = value;
}

int main(int argc, char* argv[])
{
    assert(argc==4);
    //B = atoi(argv[3]);

    //cudaStreamCreate(&stream[0]);

    gettimeofday(&start_time, NULL);

    input(argv[1]);
    gettimeofday(&temp_time, NULL);
    printf("input> %g s\n", CAL_TIME);

    //print();
    block_FW();
    gettimeofday(&temp_time, NULL);
    printf("block_FW> %g s\n", CAL_TIME);

    //print();
    output(argv[2]);
    gettimeofday(&temp_time, NULL);
    printf("output> %g s\n", CAL_TIME);

    //cudaStreamDestroy(stream[0]);

    return 0;
}


void block_FW()
{
    dim3 blocks_p3(Rounds, Rounds);
    dim3 threads(block_size, block_size);

    for (int r = 0; r < Rounds; ++r) {
        int bstart = r * block_size;
        //int block_round = MIN(n-bstart, B);
        int pivot = bstart * pitch_int + bstart;

        // Phase 1
        cal_phase1<<<1, threads>>>(pivot);

        // Phase 2
        cal_phase2_row<<<Rounds, threads>>>(r);
        cal_phase2_col<<<Rounds, threads>>>(r);

        // Phase 3
        cal_phase3<<<blocks_p3, threads>>>(r);

    }
    set_inf<<<blocks_p3, threads>>>();
    cudaMemcpy2DAsync(Dist_h, n_bytes, Dist_d, pitch_d, n_bytes, n, cudaMemcpyDeviceToHost);
}

void input(char *inFileName)
{
    FILE *infile = fopen(inFileName, "rb");
    fseek(infile, 0L, SEEK_END);
    size_t sz = ftell(infile);
    fseek(infile, 0L, SEEK_SET);

    if(sz<INF)
        fstr = buf;
    else
        fstr = (char *) malloc(sz+10);

    size_t fsize = fread(fstr, sizeof(char), sz+10, infile);
    char *tok, *next_tok;
    size_t m;

    fstr[fsize] = '\0';
    tok = strtok_r(fstr, " ", &next_tok);
    n = atoi(tok);
    tok = strtok_r(NULL, "\n", &next_tok);
    m = atoi(tok);

    Rounds = CEIL(n, block_size);
    b_rounds = block_size * Rounds;
    b_rounds_bytes = b_rounds * sizeof(int);
    dist_size = b_rounds * b_rounds;
    dist_size_bytes = dist_size * sizeof(int);
    n_bytes = n * sizeof(int);

    cudaMallocPitch(&Dist_d, &pitch_d, b_rounds_bytes, b_rounds);
    pitch_int = pitch_d / sizeof(int);
    cudaMemcpyToSymbolAsync(Dist, &Dist_d, sizeof(Dist_d), 0);
    cudaMemcpyToSymbolAsync(pitch, &pitch_int, sizeof(pitch_int), 0);
    cudaMallocHost(&Dist_h, dist_size_bytes);
    memset(Dist_h, 64, dist_size_bytes);

    for (int i = 0; i < dist_size; i+=b_rounds+1) {
        Dist_h[i] = 0;
    }
    gettimeofday(&temp_time, NULL);
    printf("\tbefore> %g s\n", CAL_TIME);

    split_strings(m, next_tok);

    gettimeofday(&temp_time, NULL);
    printf("\tafter> %g s\n", CAL_TIME);

    cudaMemcpy2DAsync(Dist_d, pitch_d, Dist_h, b_rounds_bytes, b_rounds_bytes, b_rounds, cudaMemcpyHostToDevice);

    if(sz>=INF)
        free(fstr);

    fclose(infile);
}

void output(char *outFileName)
{
    FILE *outfile = fopen(outFileName, "w");
    ftruncate(fileno(outfile), n*n);

    cudaDeviceSynchronize();
    fwrite(Dist_h, sizeof(int), n*n, outfile);

    cudaFree(Dist_d);
    cudaFreeHost(Dist_h);

    fclose(outfile);
}

void print()
{
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            int k = i * b_rounds + j;
            if(Dist_h[k]==INF)
                printf("INF ");
            else
                printf("%3d ", Dist_h[k]);
        }
        printf("\n");
    }
    printf("\n");
}

void split_strings(int m, char *fstrp)
{
    int a, b, v;
    while(m-->0) {

        if(fstrp[1]==' ') {
            a = C2I(0);
            fstrp += 2;
        } else if(fstrp[2]==' ') {
            a = C2I(0) * 10 + C2I(1);
            fstrp += 3;
        } else if(fstrp[3]==' ') {
            a = C2I(0) * 100 + C2I(1) * 10 + C2I(2);
            fstrp += 4;
        } else if(fstrp[4]==' ') {
            a = C2I(0) * 1000 + C2I(1) * 100 + C2I(2) * 10 + C2I(3);
            fstrp += 5;
        } else {
            a = C2I(0) * 10000 + C2I(1) * 1000 + C2I(2) * 100 + C2I(3) * 10 + C2I(4);
            fstrp += 6;
        }

        if(fstrp[1]==' ') {
            b = C2I(0);
            fstrp += 2;
        } else if(fstrp[2]==' ') {
            b = C2I(0) * 10 + C2I(1);
            fstrp += 3;
        } else if(fstrp[3]==' ') {
            b = C2I(0) * 100 + C2I(1) * 10 + C2I(2);
            fstrp += 4;
        } else if(fstrp[4]==' ') {
            b = C2I(0) * 1000 + C2I(1) * 100 + C2I(2) * 10 + C2I(3);
            fstrp += 5;
        } else {
            b = C2I(0) * 10000 + C2I(1) * 1000 + C2I(2) * 100 + C2I(3) * 10 + C2I(4);
            fstrp += 6;
        }

        if(fstrp[1]=='\n') {
            v = C2I(0);
            fstrp += 2;
        } else if(fstrp[2]=='\n') {
            v = C2I(0) * 10 + C2I(1);
            fstrp += 3;
        } else {
            v = C2I(0) * 100 + C2I(1) * 10 + C2I(2);
            fstrp += 4;
        }
        Dist_h[ b_rounds * a + b ] = v;
    }
}
