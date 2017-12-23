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
#define ROW_COL(__i) ( __i / line_d ), ( ( __i % pitch ) / block_size )

#define INF 1000000000
#define V   20010

void input(char *inFileName);
void output(char *outFileName);
void block_FW();
void block_FW_S();
void print();
void split_strings(int m, char *sstart);
void cuda_init();

int n, n_bytes; // Number of vertices, edges
int Rounds, b_rounds, b_rounds_bytes, dist_size, dist_size_bytes;
int diag_size;
unsigned int *Dist_h;
char buf[INF];
char *fstr;

struct timeval start_time, temp_time;

unsigned int *Dist_d;
int pitch_bytes, pitch;
__constant__ unsigned int *Dist;
__constant__ int pitch_d;

//const int tpb = 1024;
const int block_size = 32;
const int max_streams = 4;
const int first_round = 4;

dim3 threads(block_size, block_size);
size_t line_h = block_size * b_rounds;
size_t line_d = block_size * pitch;
size_t line_n = block_size * n;

cudaStream_t stream[max_streams], stream_s;
cudaEvent_t ev_before, ev_2;


__global__ void cal_phase1(int pivot)
{
    __shared__ unsigned int block_dist[block_size][block_size];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * pitch_d + tx;
    int block_index = pivot + tid;

    block_dist[ty][tx] = Dist[block_index];
    __syncthreads();

    for(int k=0; k<block_size; k++) {
        unsigned int new_dist = block_dist[ty][k] + block_dist[k][tx];
        if (block_dist[ty][tx] > new_dist)
            block_dist[ty][tx] = new_dist;
        __syncthreads();
    }

    Dist[block_index] = block_dist[ty][tx];
}

__global__ void cal_phase2_row(int pivot, int r)
{
    __shared__ unsigned int block_dist[block_size][block_size];
    __shared__ unsigned int pivot_dist[block_size][block_size];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * pitch_d + tx;
    int column = block_size * (blockIdx.x - r);
    int block_index, pivot_index;

    pivot_index = pivot + tid;

    if(blockIdx.x==r) // pivot block
        return;

    pivot_dist[ty][tx] = Dist[pivot_index];

    block_index = pivot_index + column;
    block_dist[ty][tx] = Dist[block_index];
    __syncthreads();

    for(int k=0; k<block_size; k++) {
        unsigned int new_dist = pivot_dist[ty][k] + block_dist[k][tx];

        if (block_dist[ty][tx] > new_dist)
            block_dist[ty][tx] = new_dist;
        __syncthreads();
    }

    Dist[block_index] = block_dist[ty][tx];
}

__global__ void cal_phase2_blk(int p1_pivot, int p2_pivot)
{
    __shared__ unsigned int block_dist[block_size][block_size];
    __shared__ unsigned int pivot_dist[block_size][block_size];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * pitch_d + tx;
    int block_index;

    pivot_dist[ty][tx] = Dist[p1_pivot + tid];

    block_index = p2_pivot + tid;
    block_dist[ty][tx] = Dist[block_index];
    __syncthreads();

    for(int k=0; k<block_size; k++) {
        unsigned int new_dist = block_dist[ty][k] + pivot_dist[k][tx];

        if (block_dist[ty][tx] > new_dist)
            block_dist[ty][tx] = new_dist;
        __syncthreads();
    }

    Dist[block_index] = block_dist[ty][tx];
}

__global__ void cal_phase2_col(int pivot, int r)
{
    __shared__ unsigned int block_dist[block_size][block_size];
    __shared__ unsigned int pivot_dist[block_size][block_size];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * pitch_d + tx;
    int block_index, pivot_index;
    int row_diff = pitch_d * block_size * (blockIdx.x - r);

    pivot_index = pivot + tid;

    if(row_diff==0) // pivot
        return;

    pivot_dist[tx][ty] = Dist[pivot_index];

    block_index = pivot_index + row_diff;
    block_dist[ty][tx] = Dist[block_index];
    __syncthreads();

    for(int k=0; k<block_size; k++) {
        unsigned int new_dist = block_dist[ty][k] + pivot_dist[tx][k];

        if (block_dist[ty][tx] > new_dist)
            block_dist[ty][tx] = new_dist;
        __syncthreads();
    }

    Dist[block_index] = block_dist[ty][tx];
}

__global__ void cal_phase3(int p1_pivot, int p2_pivot, int r)
{
    __shared__ unsigned int pvRow_dist[block_size][block_size];
    __shared__ unsigned int pvCol_dist[block_size][block_size];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * pitch_d + tx;
    int col_diff = (blockIdx.x - r) * block_size;
    int block_index, p1_index, p2_index;
    unsigned int block_dist;

    p1_index = p1_pivot + col_diff + tid;
    p2_index = p2_pivot + tid;

    if(col_diff==0) // pivots
        return;

    pvRow_dist[ty][tx] = Dist[p1_index];
    pvCol_dist[ty][tx] = Dist[p2_index];
    __syncthreads();

    block_index = p2_index + col_diff;
    block_dist = Dist[block_index];

    for(int k=0; k<block_size; k++) {
        unsigned int new_dist = pvCol_dist[ty][k] + pvRow_dist[k][tx];
        if (block_dist > new_dist)
            block_dist = new_dist;
    }

    Dist[block_index] = block_dist;
}

__global__ void cal_phase2_blk_2(int p1_pivot, int p2_pivot)
{
    __shared__ unsigned int block_dist[block_size][block_size];
    __shared__ unsigned int pivot_dist[block_size][block_size];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * pitch_d + tx;
    int block_index;

    pivot_dist[ty][tx] = Dist[p1_pivot + tid];

    block_index = p2_pivot + tid + blockIdx.x * pitch_d * block_size;
    block_dist[ty][tx] = Dist[block_index];
    __syncthreads();

    for(int k=0; k<block_size; k++) {
        unsigned int new_dist = block_dist[ty][k] + pivot_dist[k][tx];

        if (block_dist[ty][tx] > new_dist)
            block_dist[ty][tx] = new_dist;
        __syncthreads();
    }

    Dist[block_index] = block_dist[ty][tx];
}

__global__ void cal_phase3_2(int p1_pivot, int p2_pivot, int p3_pivot, int r)
{
    __shared__ unsigned int pvR1_dist[block_size][block_size];
    __shared__ unsigned int pvC1_dist[block_size][block_size];
    __shared__ unsigned int pvC2_dist[block_size][block_size];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * pitch_d + tx;
    int col_diff = (blockIdx.x - r) * block_size;
    int b1_index, b2_index, p1_index, p2_index, p3_index;
    unsigned int b1_dist, b2_dist, inter[block_size];

    p1_index = p1_pivot + tid + col_diff;
    p2_index = p2_pivot + tid;
    p3_index = p3_pivot + tid;

    if(col_diff==0) // pivots
        return;

    pvR1_dist[ty][tx] = Dist[p1_index];
    pvC1_dist[ty][tx] = Dist[p2_index];
    pvC2_dist[ty][tx] = Dist[p3_index];
    __syncthreads();

    b1_index = p2_index + col_diff;
    b2_index = p3_index + col_diff;
    b1_dist = Dist[b1_index];
    b2_dist = Dist[b2_index];

    for(int k=0; k<block_size; k++) {
        inter[k] = pvR1_dist[k][tx];
        unsigned int new_dist = pvC1_dist[ty][k] + inter[k];
        if (b1_dist > new_dist)
            b1_dist = new_dist;
    }
    Dist[b1_index] = b1_dist;

    for(int k=0; k<block_size; k++) {
        unsigned int new_dist = pvC2_dist[ty][k] + inter[k];
        if (b2_dist > new_dist)
            b2_dist = new_dist;
    }
    Dist[b2_index] = b2_dist;
}

__global__ void set_inf_row(unsigned int *ptr_d)
{
    int dist_index = pitch_d * threadIdx.y + block_size * blockIdx.x + threadIdx.x;
    unsigned int value = ptr_d[dist_index];

    if (value>=INF)
        value = INF;

    ptr_d[dist_index] = value;
}

int main(int argc, char* argv[])
{
    assert(argc==4);
    //block_size = atoi(argv[3]);

    gettimeofday(&start_time, NULL);

    input(argv[1]);
    gettimeofday(&temp_time, NULL);
    printf("input> %g s\n", CAL_TIME);

    cuda_init();

    if(Rounds<=first_round)
        block_FW_S();
    else
        block_FW();

    gettimeofday(&temp_time, NULL);
    printf("block_FW> %g s\n", CAL_TIME);

    output(argv[2]);
    gettimeofday(&temp_time, NULL);
    printf("output> %g s\n", CAL_TIME);

    return 0;
}


void cuda_init()
{
    cudaMemcpy2DAsync(Dist_d, pitch_bytes, Dist_h, b_rounds_bytes, b_rounds_bytes, block_size, cudaMemcpyHostToDevice);

    line_h = block_size * b_rounds;
    line_d = block_size * pitch;
    line_n = block_size * n;
    diag_size = (pitch + 1) * block_size;

    //cudaStreamCreate(&stream_s1);
    cudaStreamCreate(&stream[0]);
    cal_phase1<<<1, threads, 0, stream[0]>>>(0);
    cal_phase2_row<<<Rounds, threads, 0, stream[0]>>>(0, 0);

    cudaEventCreate(&ev_before);
    cudaEventCreate(&ev_2);

}

void block_FW()
{
    int p1_start = 0, p2_start = 0;
    unsigned int *ptr_h = Dist_h, *ptr_d = Dist_d;
    int p2_sub;
    int s = 0;
    int flag;

    cudaEventRecord(ev_2, stream[0]);
    //printf("round 1\n");
    for(int i=1; i<first_round; i++) {
        ptr_h += line_h;
        ptr_d += line_d;
        cudaStreamCreate(&stream[i]);
        cudaMemcpy2DAsync(ptr_d, pitch_bytes, ptr_h, b_rounds_bytes, b_rounds_bytes, block_size, cudaMemcpyHostToDevice, stream[i]);
        cudaStreamWaitEvent(stream[i], ev_2, 0);

        p2_start += line_d;
        //printf("\tp1=(%d,%d), p2=(%d,%d) stream %d\n", ROW_COL(p1_start), ROW_COL(p2_start), i);
        cal_phase2_blk<<< 1, threads, 0, stream[i]>>>(p1_start, p2_start);
        cal_phase3<<<Rounds, threads, 0, stream[i]>>>(p1_start, p2_start, 0);
    }
    for(int i=1; i<first_round; i++) {
        p1_start += diag_size;
        //printf("round %d: p1=(%d,%d) stream %d\n", i, ROW_COL(p1_start), i);
        cal_phase1<<<1, threads, 0, stream[i]>>>(p1_start);
        cal_phase2_row<<<Rounds, threads, 0, stream[i]>>>(p1_start, i);

        cudaEventRecord(ev_2, stream[i]);
        for(int j=0; j<first_round; j++) {
            if(i==j) continue;

            cudaStreamWaitEvent(stream[j], ev_2, 0);

            p2_sub = p1_start + line_d * (j - i);
            //printf("\tp1=(%d,%d), p2=(%d,%d) stream %d\n", ROW_COL(p1_start), ROW_COL(p2_sub), j);
            cal_phase2_blk<<< 1, threads, 0, stream[j]>>>(p1_start, p2_sub);
            cal_phase3<<<Rounds, threads, 0, stream[j]>>>(p1_start, p2_sub, i);
        }
    }

    for(int i=first_round; i<Rounds; i++) {

        ptr_h += line_h;
        ptr_d += line_d;
        cudaMemcpy2DAsync(ptr_d, pitch_bytes, ptr_h, b_rounds_bytes, b_rounds_bytes, block_size, cudaMemcpyHostToDevice, stream[s]);

        p2_start += line_d;

        p1_start = 0;
        p2_sub = p2_start;
        //printf("\nround %d seq\n", i);
        for(int r=0; r<first_round; r++) {
            //printf("\tp1=(%d,%d), p2=(%d,%d) stream %d\n", ROW_COL(p1_start), ROW_COL(p2_sub), s);

            cal_phase2_blk<<< 1, threads, 0, stream[s]>>>(p1_start, p2_sub);
            cal_phase3<<<Rounds, threads, 0, stream[s]>>>(p1_start, p2_sub, r);
            p1_start += diag_size;
            p2_sub += block_size;
        }
        if(i==first_round) {
            cudaStreamCreate(&stream_s);
            cudaEventRecord(ev_before, stream[s]);
            cudaStreamWaitEvent(stream_s, ev_before, 0);
        }

        s = s+1<max_streams ? s+1 : 0;
    }

    s = 0;
    //printf("R %d\n", Rounds);
    for (int r=first_round; r<Rounds; ++r) {

        cal_phase1<<<1, threads, 0, stream_s>>>(p1_start);

        cal_phase2_row<<<Rounds, threads, 0, stream_s>>>(p1_start, r);
        cudaEventRecord(ev_2, stream_s);
        for(int i=0; i<max_streams; i++)
            cudaStreamWaitEvent(stream[i], ev_2, 0);

        if(r==Rounds-1) {
            ptr_h = Dist_h + r * line_n;
            ptr_d = Dist_d + r * line_d;
            set_inf_row<<<Rounds, threads, 0, stream_s>>>(ptr_d);
            cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, block_size, cudaMemcpyDeviceToHost, stream_s);
        }

        //printf("r %d\n", r);
        flag = 1;
        for(int i = (r+1) % Rounds; i != r; i = (i==Rounds-1) ? 0 : i+1) {
            p2_start = p1_start + line_d * (i-r);
/*
            cal_phase2_blk<<< 1, threads, 0, stream[s]>>>(p1_start, p2_start);
            cal_phase3<<<Rounds, threads, 0, stream[s]>>>(p1_start, p2_start, r);
*/
            if(flag) {
                if(i==Rounds-1 || i==r-1) {
                    //printf("\ti %d\n", i);
                    cal_phase2_blk<<< 1, threads, 0, stream[s]>>>(p1_start, p2_start);
                    cal_phase3<<<Rounds, threads, 0, stream[s]>>>(p1_start, p2_start, r);
                } else {

                    //printf("\ti %d %d\n", i, (i+1)%Rounds);
                    cal_phase2_blk_2<<< 2, threads, 0, stream[s]>>>(p1_start, p2_start);
                    cal_phase3_2<<<Rounds, threads, 0, stream[s]>>>(p1_start, p2_start, p2_start + line_d, r);
                    flag = 0;
/*
                    cal_phase2_blk_2<<< 2, threads, 0, stream[s]>>>(p1_start, p2_start);
                    cal_phase3<<<Rounds, threads, 0, stream[s]>>>(p1_start, p2_start, r);
                    cal_phase3<<<Rounds, threads, 0, stream[s]>>>(p1_start, p2_start+line_d, r);
*/
                    flag = 0;
                }
            } else {
                flag = 1;
            }

            if(i==r+1) {
                cudaEventRecord(ev_before, stream[s]);
                cudaStreamWaitEvent(stream_s, ev_before, 0);
            }

            if(r==Rounds-1) {
                ptr_d = Dist_d + i * line_d;
                set_inf_row<<<Rounds, threads, 0, stream[s]>>>(ptr_d);
                ptr_h = Dist_h + i * line_n;
                cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, block_size, cudaMemcpyDeviceToHost, stream[s]);
                if(!flag) {
                    ptr_d += line_d;
                    set_inf_row<<<Rounds, threads, 0, stream[s]>>>(ptr_d);
                    ptr_h += line_n;
                    cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, block_size, cudaMemcpyDeviceToHost, stream[s]);
                }
            }
            if(flag)
                s = s+1<max_streams ? s+1 : 0;
        }
        p1_start += diag_size;
    }
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
    size_t m, p_bytes;

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

    cudaMallocPitch(&Dist_d, &p_bytes, b_rounds_bytes, b_rounds);
    pitch_bytes = p_bytes;
    pitch = pitch_bytes / sizeof(int);
    cudaMemcpyToSymbolAsync(Dist, &Dist_d, sizeof(Dist_d), 0);
    cudaMemcpyToSymbolAsync(pitch_d, &pitch, sizeof(pitch), 0);
    cudaMallocHost(&Dist_h, dist_size_bytes);
    memset(Dist_h, 64, dist_size_bytes);

    for (int i = 0; i < dist_size; i+=b_rounds+1) {
        Dist_h[i] = 0;
    }
    gettimeofday(&temp_time, NULL);
    //printf("\tbefore> %g s\n", CAL_TIME);

    split_strings(m, next_tok);

    gettimeofday(&temp_time, NULL);
    //printf("\tafter> %g s\n", CAL_TIME);


    if(sz>=INF)
        free(fstr);

    fclose(infile);
}

void output(char *outFileName)
{
    FILE *outfile = fopen(outFileName, "w");
    ftruncate(fileno(outfile), n*n);

    for(int i=0; i<max_streams; i++) {
        cudaStreamDestroy(stream[i]);
    }
    cudaStreamDestroy(stream_s);
    cudaEventDestroy(ev_before);
    cudaEventDestroy(ev_2);

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

void block_FW_S()
{
    int p1_start = 0, p2_start = 0;
    unsigned int *ptr_h = Dist_h, *ptr_d = Dist_d;
    int p2_sub;

    cudaEventRecord(ev_2, stream[0]);
    //printf("round 1\n");
    for(int i=1; i<Rounds; i++) {
        ptr_h += line_h;
        ptr_d += line_d;
        cudaStreamCreate(&stream[i]);
        cudaMemcpy2DAsync(ptr_d, pitch_bytes, ptr_h, b_rounds_bytes, b_rounds_bytes, block_size, cudaMemcpyHostToDevice, stream[i]);
        cudaStreamWaitEvent(stream[i], ev_2, 0);

        p2_start += line_d;
        //printf("\tp1=(%d,%d), p2=(%d,%d) stream %d\n", ROW_COL(p1_start), ROW_COL(p2_start), i);
        cal_phase2_blk<<< 1, threads, 0, stream[i]>>>(p1_start, p2_start);
        cal_phase3<<<Rounds, threads, 0, stream[i]>>>(p1_start, p2_start, 0);
    }
    for(int i=1; i<Rounds; i++) {
        p1_start += diag_size;
        //printf("round %d: p1=(%d,%d) stream %d\n", i, ROW_COL(p1_start), i);
        cal_phase1<<<1, threads, 0, stream[i]>>>(p1_start);
        cal_phase2_row<<<Rounds, threads, 0, stream[i]>>>(p1_start, i);

        cudaEventRecord(ev_2, stream[i]);
        for(int j=0; j<Rounds; j++) {
            if(i==j) continue;

            cudaStreamWaitEvent(stream[j], ev_2, 0);

            p2_sub = p1_start + line_d * (j - i);
            //printf("\tp1=(%d,%d), p2=(%d,%d) stream %d\n", ROW_COL(p1_start), ROW_COL(p2_sub), j);
            cal_phase2_blk<<< 1, threads, 0, stream[j]>>>(p1_start, p2_sub);
            cal_phase3<<<Rounds, threads, 0, stream[j]>>>(p1_start, p2_sub, i);
        }
    }

    ptr_h = Dist_h;
    ptr_d = Dist_d;
    for(int i=0; i<Rounds; i++) {
        set_inf_row<<<Rounds, threads, 0, stream[i]>>>(ptr_d);
        cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, block_size, cudaMemcpyDeviceToHost, stream[i]);
        ptr_h += line_n;
        ptr_d += line_d;
    }
    cudaStreamCreate(&stream_s);
}
