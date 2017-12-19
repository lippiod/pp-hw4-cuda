#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

#define CEIL(a, b) ( ((a) + (b) - 1) / (b) )
#define MIN(a, b) ( (a) < (b) ? (a) : (b) )
#define CAL_TIME ( 1e-6 * (temp_time.tv_usec - start_time.tv_usec) + (temp_time.tv_sec - start_time.tv_sec) )

#define INF 1000000000
#define V   20500

void input(char *inFileName);
void output(char *outFileName);
void block_FW();
void print();

int n, n_bytes;	// Number of vertices, edges
int B;
int Rounds, b_rounds, b_rounds_bytes, dist_size, dist_size_bytes;
unsigned int *Dist_h;
char buf[INF];
char *fstr;

struct timeval start_time, temp_time;

unsigned int *Dist_d;
size_t pitch;

//const int tpb = 1024;
const int maxBF = 40;

__global__ void cal_phase1(unsigned int *Dist_d, size_t pitch, int round, int pivot)
{
    __shared__ unsigned int block_dist[maxBF][maxBF];

    int tidx = threadIdx.x, tidy = threadIdx.y;
    int block_index = pitch * (pivot + tidy) + pivot + tidx;

    block_dist[tidy][tidx] = Dist_d[block_index];
    __syncthreads();

    for(int k=0; k<round; k++) {
        unsigned int new_dist = block_dist[tidy][k] + block_dist[k][tidx];
        if (block_dist[tidy][tidx] > new_dist)
            block_dist[tidy][tidx] = new_dist;
        __syncthreads();
    }

    Dist_d[block_index] = block_dist[tidy][tidx];
}

__global__ void cal_phase2(unsigned int *Dist_d, size_t pitch, int total, int round, int r)
{
    __shared__ unsigned int block_dist[maxBF][maxBF];
    __shared__ unsigned int pivot_dist[maxBF][maxBF];

    int tidx = threadIdx.x, tidy = threadIdx.y;
    int pivot = blockDim.x * r;
    int block_index, pivot_index;

    if(blockIdx.x==r) // pivot block
        return;

    pivot_index = pitch * (pivot + tidy) + pivot + tidx;
    pivot_dist[tidy][tidx] = Dist_d[pivot_index];

    if(blockIdx.y==0) // row pivot
        block_index = pitch * (pivot + tidy) + blockDim.x * blockIdx.x + tidx;
    else // column pivot
        block_index = pitch * (blockDim.x * blockIdx.x + tidy) + pivot + tidx;
    block_dist[tidy][tidx] = Dist_d[block_index];
    __syncthreads();

    for(int k=0; k<round; k++) {
        unsigned int new_dist;
        if(blockIdx.y==0)
            new_dist = pivot_dist[tidy][k] + block_dist[k][tidx];
        else
            new_dist = block_dist[tidy][k] + pivot_dist[k][tidx];

        if (block_dist[tidy][tidx] > new_dist)
            block_dist[tidy][tidx] = new_dist;
        __syncthreads();
    }

    Dist_d[block_index] = block_dist[tidy][tidx];
}

__global__ void cal_phase3(unsigned int *Dist_d, size_t pitch, int total, int round, int r)
{
    __shared__ unsigned int block_dist[maxBF][maxBF];
    __shared__ unsigned int pvRow_dist[maxBF][maxBF];
    __shared__ unsigned int pvCol_dist[maxBF][maxBF];

    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bx = blockDim.x * blockIdx.x, by = blockDim.y * blockIdx.y;
    int pv = blockDim.x * r;
    int block_index, pvRow_index, pvCol_index;

    if(blockIdx.y==r || blockIdx.x==r) // pivots
        return;

    block_index = pitch * (by + tidy) + bx + tidx;
    block_dist[tidy][tidx] = Dist_d[block_index];

    pvRow_index = pitch * (pv + tidy) + bx + tidx;
    pvRow_dist[tidy][tidx] = Dist_d[pvRow_index];

    pvCol_index = pitch * (by + tidy) + pv + tidx;
    pvCol_dist[tidy][tidx] = Dist_d[pvCol_index];
    __syncthreads();

    for(int k=0; k<round; k++) {
        unsigned int new_dist = pvCol_dist[tidy][k] + pvRow_dist[k][tidx];
        if (block_dist[tidy][tidx] > new_dist)
            block_dist[tidy][tidx] = new_dist;
        __syncthreads();
    }

    Dist_d[block_index] = block_dist[tidy][tidx];
}

__global__ void set_inf(unsigned int *Dist_d, int pitch)
{
    int dist_index = pitch * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int value = Dist_d[ dist_index ];

    if (value>=INF)
        value = INF;

    Dist_d[ dist_index ] = value;
}

int main(int argc, char* argv[])
{
    assert(argc==4);
	B = atoi(argv[3]);

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

	return 0;
}


void block_FW()
{
    dim3 blocks_p2(Rounds, 2);
    dim3 blocks_p3(Rounds, Rounds);
    dim3 threads(B, B);
    int pitch_int = pitch / sizeof(int);

	for (int r = 0; r < Rounds; ++r) {
        int bstart = r * B;
        int block_round = MIN(n-bstart, B);

        //printf("%d %d\n", r, round);
		// Phase 1
        cal_phase1<<<1, threads>>>(Dist_d, pitch_int, block_round, bstart);

		// Phase 2
        cal_phase2<<<blocks_p2, threads>>>(Dist_d, pitch_int, Rounds, block_round, r);

        // Phase 3
        cal_phase3<<<blocks_p3, threads>>>(Dist_d, pitch_int, Rounds, block_round, r);
	}

    set_inf<<<blocks_p3, threads>>>(Dist_d, pitch_int);
    //print();
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
    int m;

    fstr[fsize] = '\0';
    tok = strtok_r(fstr, " ", &next_tok);
    n = atoi(tok);
    tok = strtok_r(NULL, "\n", &next_tok);
    m = atoi(tok);

    Rounds = CEIL(n, B);
    b_rounds = B * Rounds;
    b_rounds_bytes = b_rounds * sizeof(int);
    dist_size = b_rounds * b_rounds;
    dist_size_bytes = dist_size * sizeof(int);
    n_bytes = n * sizeof(int);

    cudaMallocPitch((void **) &Dist_d, &pitch, b_rounds_bytes, b_rounds);
    cudaMallocHost((void **) &Dist_h, dist_size_bytes);
    memset(Dist_h, 64, dist_size_bytes);

	for (int i = 0; i < dist_size; i+=b_rounds+1) {
        Dist_h[i] = 0;
	}

    gettimeofday(&temp_time, NULL);
    printf("\tbefore> %g s\n", CAL_TIME);
	while (--m >= 0) {
		int a, b, v;
        tok = strtok_r(NULL, " ", &next_tok);
        a = atoi(tok);
        tok = strtok_r(NULL, " ", &next_tok);
        b = atoi(tok);
        tok = strtok_r(NULL, "\n", &next_tok);
        v = atoi(tok);
		Dist_h[ b_rounds * a + b ] = v;
	}
    gettimeofday(&temp_time, NULL);
    printf("\tafter> %g s\n", CAL_TIME);

    cudaMemcpy2D(Dist_d, pitch, Dist_h, b_rounds_bytes, b_rounds_bytes, b_rounds, cudaMemcpyHostToDevice);

    if(sz>=INF)
        free(fstr);

    fclose(infile);
}

void output(char *outFileName)
{
    cudaMemcpy2D(Dist_h, n_bytes, Dist_d, pitch, n_bytes, n, cudaMemcpyDeviceToHost);
    cudaFree(Dist_d);
    //print();

    gettimeofday(&temp_time, NULL);
    printf("\tbefore> %g s\n", CAL_TIME);

	FILE *outfile = fopen(outFileName, "w");
    fwrite(Dist_h, sizeof(int), n*n, outfile);

    cudaFreeHost(Dist_h);

    gettimeofday(&temp_time, NULL);
    printf("\tafter> %g s\n", CAL_TIME);

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
