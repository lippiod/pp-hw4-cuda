#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

#define CEIL(a, b) ( ((a) + (b) - 1) / (b) )
#define MIN(a, b) ( (a) < (b) ? (a) : (b) )
#define INF 1000000000
#define V   20500

void input(char *inFileName);
void output(char *outFileName);
void block_FW();
void cuda_init();
void cuda_fin();
void print();

int n;	// Number of vertices, edges
int B;
int Rounds;
static int Dist[V][V];


int *Dist_d;
size_t pitch;

//const int tpb = 1024;
const int maxBF = 40;

__global__ void cal_phase1(int *Dist_d, size_t pitch, int round, int pivot)
{
    __shared__ int block_dist[maxBF][maxBF];

    int tidx = threadIdx.x, tidy = threadIdx.y;
    int block_index = pitch * (pivot + tidy) + pivot + tidx;

    block_dist[tidy][tidx] = Dist_d[block_index];
    __syncthreads();

    for(int k=0; k<round; k++) {
        int new_dist = block_dist[tidy][k] + block_dist[k][tidx];
        if (block_dist[tidy][tidx] > new_dist)
            block_dist[tidy][tidx] = new_dist;
        __syncthreads();
    }

    Dist_d[block_index] = block_dist[tidy][tidx];
}

__global__ void cal_phase2(int *Dist_d, size_t pitch, int total, int round, int r, int B)
{
    __shared__ int block_dist[maxBF][maxBF];
    __shared__ int pivot_dist[maxBF][maxBF];

    int tidx = threadIdx.x, tidy = threadIdx.y;
    int pivot = B * r;
    int block_index, pivot_index;

    if(blockIdx.x==r) // pivot block
        return;

    pivot_index = pitch * (pivot + tidy) + pivot + tidx;
    pivot_dist[tidy][tidx] = Dist_d[pivot_index];

    if(blockIdx.y==0) // row pivot
        block_index = pitch * (pivot + tidy) + B * blockIdx.x + tidx;
    else // column pivot
        block_index = pitch * (B * blockIdx.x + tidy) + pivot + tidx;
    block_dist[tidy][tidx] = Dist_d[block_index];
    __syncthreads();

    for(int k=0; k<round; k++) {
        int new_dist;
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

__global__ void cal_phase3(int *Dist_d, size_t pitch, int total, int round, int r, int B)
{
    __shared__ int block_dist[maxBF][maxBF];
    __shared__ int pvRow_dist[maxBF][maxBF];
    __shared__ int pvCol_dist[maxBF][maxBF];

    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bx = B * blockIdx.x, by = B * blockIdx.y;
    int pv = B * r;
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
        int new_dist = pvCol_dist[tidy][k] + pvRow_dist[k][tidx];
        if (block_dist[tidy][tidx] > new_dist)
            block_dist[tidy][tidx] = new_dist;
        __syncthreads();
    }

    Dist_d[block_index] = block_dist[tidy][tidx];
}

int main(int argc, char* argv[])
{
    assert(argc==4);
	B = atoi(argv[3]);

	input(argv[1]);

    Rounds = CEIL(n, B);
    //print();
    cuda_init();
	block_FW();
    cuda_fin();
    //print();

	output(argv[2]);

	return 0;
}


void block_FW()
{
    dim3 blocks_p2(Rounds, 2);
    dim3 blocks_p3(Rounds, Rounds);
    dim3 threads(B, B);

	for (int r = 0; r < Rounds; ++r) {
        int bstart = r * B;
        int pitch_int = pitch / sizeof(int);
        int block_round = MIN(n-bstart, B);

        //printf("%d %d\n", r, round);
		// Phase 1
        cal_phase1<<<1, threads>>>(Dist_d, pitch_int, block_round, bstart);

		// Phase 2
        cal_phase2<<<blocks_p2, threads>>>(Dist_d, pitch_int, Rounds, block_round, r, B);

        // Phase 3
        cal_phase3<<<blocks_p3, threads>>>(Dist_d, pitch_int, Rounds, block_round, r, B);
	}
}

void cuda_init()
{
    //cudaError_t ce;
    cudaMallocPitch((void **) &Dist_d, &pitch, B*Rounds*sizeof(int), B*Rounds);
    //fprintf(stderr, "%s\n", cudaGetErrorString(ce));
    cudaMemcpy2D(Dist_d, pitch, Dist, V*sizeof(int), n*sizeof(int), n, cudaMemcpyHostToDevice);
    //fprintf(stderr, "%s\n", cudaGetErrorString(ce));
}

void cuda_fin()
{
    cudaMemcpy2D(Dist, V*sizeof(int), Dist_d, pitch, n*sizeof(int), n, cudaMemcpyDeviceToHost);
    cudaFree(Dist_d);
}

void input(char *inFileName)
{
    int m;
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", &n, &m);

	for (int i = 0; i < V; ++i) {
		for (int j = 0; j < V; ++j) {
			if (i == j)	Dist[i][j] = 0;
			else		Dist[i][j] = INF;
		}
	}

	while (--m >= 0) {
		int a, b, v;
		fscanf(infile, "%d %d %d", &a, &b, &v);
		Dist[a][b] = v;
	}
    fclose(infile);
}

void output(char *outFileName)
{
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF)
                Dist[i][j] = INF;
		}
		fwrite(Dist[i], sizeof(int), n, outfile);
	}
    fclose(outfile);
}

void print()
{
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            if(Dist[i][j]==INF)
                printf("INF ");
            else
                printf("%3d ", Dist[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
