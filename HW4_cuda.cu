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


const int INF = 1000000000;
const int V = 20010;
//const int tpb = 1024;
const int block_size = 32;
const int file_step = 10;
const int write_size = 6;
int max_streams = 4;
int first_round = 4;

void input(char *inFileName);
void block_FW(char *outfile);
void block_FW_S(char *outFileName);
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

dim3 threads(block_size, block_size);
size_t line_h, line_d, line_n;

cudaStream_t stream[8], stream_s, stream_m;
cudaEvent_t ev_1, ev_2, ev_m;

int id[V];

__global__ void cal_phase1(int pivot)
{
    __shared__ unsigned int block_dist[block_size][block_size];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * pitch_d + tx;
    int block_index = pivot + tid;
    unsigned int origin, blk_dist, new_dist;

    block_dist[ty][tx] = origin = Dist[block_index];
    __syncthreads();

    if(origin > INF)
        Dist[block_index] = origin = INF;

    blk_dist = origin;
    for(int k=0; k<block_size-1; k++) {
        new_dist = block_dist[ty][k] + block_dist[k][tx];
        //if (block_dist[ty][tx] > new_dist)
            //block_dist[ty][tx] = new_dist;
        if(blk_dist > new_dist)
            block_dist[ty][tx] = blk_dist = new_dist;
        __syncthreads();
    }
    new_dist = block_dist[ty][block_size-1] + block_dist[block_size-1][tx];
    if(blk_dist > new_dist)
        Dist[block_index] = new_dist;
    else if(origin > blk_dist)
        Dist[block_index] = blk_dist;
}

__global__ void cal_phase2_row(int pivot, int r)
{
    __shared__ unsigned int block_dist[block_size][block_size];
    __shared__ unsigned int pivot_dist[block_size][block_size];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * pitch_d + tx;
    int column = block_size * (blockIdx.x - r);
    int block_index, pivot_index;
    unsigned int origin, blk_dist, new_dist;

    pivot_index = pivot + tid;

    if(blockIdx.x==r) // pivot block
        return;

    pivot_dist[ty][tx] = Dist[pivot_index];

    block_index = pivot_index + column;
    block_dist[ty][tx] = origin = Dist[block_index];
    __syncthreads();

    if(origin > INF)
        Dist[block_index] = origin = INF;

    blk_dist = origin;
    for(int k=0; k<block_size-1; k++) {
        new_dist = pivot_dist[ty][k] + block_dist[k][tx];

        //if (block_dist[ty][tx] > new_dist)
            //block_dist[ty][tx] = new_dist;
        if (blk_dist > new_dist)
            block_dist[ty][tx] = blk_dist = new_dist;
        __syncthreads();
    }
    new_dist = pivot_dist[ty][block_size-1] + block_dist[block_size-1][tx];
    if(blk_dist > new_dist)
        Dist[block_index] = new_dist;
    else if(origin > blk_dist)
        Dist[block_index] = blk_dist;
}

__global__ void cal_phase2_blk(int p1_pivot, int p2_pivot)
{
    __shared__ unsigned int block_dist[block_size][block_size];
    __shared__ unsigned int pivot_dist[block_size][block_size];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * pitch_d + tx;
    int block_index;
    unsigned int origin, blk_dist, new_dist;

    pivot_dist[ty][tx] = Dist[p1_pivot + tid];

    block_index = p2_pivot + tid + blockIdx.x * pitch_d * block_size;
    block_dist[ty][tx] = origin = Dist[block_index];
    __syncthreads();

    blk_dist = origin;
    for(int k=0; k<block_size-1; k++) {
        new_dist = block_dist[ty][k] + pivot_dist[k][tx];

        //if (block_dist[ty][tx] > new_dist)
            //block_dist[ty][tx] = new_dist;
        if(blk_dist > new_dist)
            block_dist[ty][tx] = blk_dist = new_dist;
        __syncthreads();
    }
    new_dist = block_dist[ty][block_size-1] + pivot_dist[block_size-1][tx];
    if(blk_dist > new_dist)
        Dist[block_index] = new_dist;
    else if(origin > blk_dist)
        Dist[block_index] = blk_dist;
}

__global__ void cal_phase3(int p1_pivot, int p2_pivot, int r)
{
    __shared__ unsigned int pvRow_dist[block_size][block_size];
    __shared__ unsigned int pvCol_dist[block_size][block_size];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * pitch_d + tx;
    int col_diff = (blockIdx.x - r) * block_size;
    int block_index, p1_index, p2_index;
    unsigned int origin, block_dist, new1, new2;

    p1_index = p1_pivot + col_diff + tid;
    p2_index = p2_pivot + tid;

    if(col_diff==0) // pivots
        return;

    pvRow_dist[ty][tx] = Dist[p1_index];
    pvCol_dist[ty][tx] = Dist[p2_index];
    __syncthreads();

    block_dist = pvCol_dist[ty][0] + pvRow_dist[0][tx];
    new1 = pvCol_dist[ty][1] + pvRow_dist[1][tx];

    block_index = p2_index + col_diff;
    origin = Dist[block_index];

    if (block_dist > new1)
        block_dist = new1;

    for(int k=2; k<block_size; k+=2) {
        new1 = pvCol_dist[ty][k] + pvRow_dist[k][tx];
        new2 = pvCol_dist[ty][k+1] + pvRow_dist[k+1][tx];
        if (block_dist > new1)
            block_dist = new1;
        if (block_dist > new2)
            block_dist = new2;
    }
    if(origin>block_dist)
        Dist[block_index] = block_dist;
    //Dist[block_index] = MIN(origin, block_dist);
}

__global__ void cal_phase3_2(int p1_pivot, int p2_pivot, int r)
{
    __shared__ unsigned int pvR1_dist[block_size][block_size];
    __shared__ unsigned int pvC1_dist[block_size][block_size];
    __shared__ unsigned int pvC2_dist[block_size][block_size];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * pitch_d + tx;
    int col_diff = (blockIdx.x - r) * block_size;
    int b1_index, b2_index, p1_index, p2_index, p3_index;
    unsigned int origin1, origin2, b1_dist, b2_dist, inter[block_size];
    unsigned int new_dist, new1, new2;

    p1_index = p1_pivot + tid + col_diff;

    if(col_diff==0) // pivots
        return;

    pvR1_dist[ty][tx] = Dist[p1_index];
    p2_index = p2_pivot + tid;
    pvC1_dist[ty][tx] = Dist[p2_index];
    p3_index = p2_index + pitch_d * block_size;
    pvC2_dist[ty][tx] = Dist[p3_index];
    __syncthreads();

    inter[0] = pvR1_dist[0][tx];
    inter[1] = pvR1_dist[1][tx];

    b1_index = p2_index + col_diff;
    b2_index = p3_index + col_diff;

    b1_dist = pvC1_dist[ty][0] + inter[0];
    new_dist = pvC1_dist[ty][1] + inter[1];

    origin1 = Dist[b1_index];
    origin2 = Dist[b2_index];

    if (b1_dist > new_dist)
        b1_dist = new_dist;

    b2_dist = pvC2_dist[ty][0] + inter[0];
    new_dist = pvC2_dist[ty][1] + inter[1];

    for(int k=2; k<block_size; k+=2) {
        inter[k] = pvR1_dist[k][tx];
        inter[k+1] = pvR1_dist[k+1][tx];

        new1 = pvC1_dist[ty][k] + inter[k];
        new2 = pvC1_dist[ty][k+1] + inter[k+1];
        if (b1_dist > new1)
            b1_dist = new1;

        if (b1_dist > new2)
            b1_dist = new2;
    }
    if(origin1>b1_dist)
        Dist[b1_index] = b1_dist;
    //Dist[b1_index] = MIN(origin1, b1_dist);

    if (b2_dist > new_dist)
        b2_dist = new_dist;
    for(int k=2; k<block_size; k+=2) {
        new1 = pvC2_dist[ty][k] + inter[k];
        new2 = pvC2_dist[ty][k+1] + inter[k+1];
        if (b2_dist > new1)
            b2_dist = new1;

        if (b2_dist > new2)
            b2_dist = new2;
    }
    if(origin2>b2_dist)
        Dist[b2_index] = b2_dist;
    //Dist[b2_index] = MIN(origin2, b2_dist);
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

    if(Rounds<=first_round) {
        block_FW_S(argv[2]);
    } else {
        block_FW(argv[2]);
    }

    gettimeofday(&temp_time, NULL);
    printf("block_FW> %g s\n", CAL_TIME);

    return 0;
}

void block_FW(char *outfile)
{
    int p1_start = 0, p2_start = 0;
    unsigned int *ptr_f = Dist_h, *ptr_h = Dist_h, *ptr_d = Dist_d;
    int p2_sub;
    int s = 0, cnt = 0;
    int flag;
    FILE *fp = fopen(outfile, "w");
    int step_size = 0;
    int total = n * n;

    id[0] = 0;
    //printf("round 1\n");
    for(int i=1; i<first_round; i++) {
        id[i] = i / 2;
        ptr_h += line_h;
        ptr_d += line_d;
        cudaMemcpy2DAsync(ptr_d, pitch_bytes, ptr_h, b_rounds_bytes, b_rounds_bytes, block_size, cudaMemcpyHostToDevice, stream_m);
        cudaEventCreateWithFlags(&ev_m, cudaEventDisableTiming);
        cudaEventRecord(ev_m, stream_m);

        cudaStreamCreate(&stream[i]);
        cudaStreamWaitEvent(stream[i], ev_1, 0);
        cudaStreamWaitEvent(stream[i], ev_m, 0);

        p2_start += line_d;
        //printf("\tp1=(%d,%d), p2=(%d,%d) stream %d\n", ROW_COL(p1_start), ROW_COL(p2_start), i);
        cal_phase2_blk<<< 1, threads, 0, stream[i]>>>(p1_start, p2_start);
        cudaStreamWaitEvent(stream[i], ev_2, 0);
        cal_phase3<<<Rounds, threads, 0, stream[i]>>>(p1_start, p2_start, 0);
    }
    for(int i=1; i<first_round; i++) {
        p1_start += diag_size;
        //printf("round %d: p1=(%d,%d) stream %d\n", i, ROW_COL(p1_start), i);
        cal_phase1<<<1, threads, 0, stream[i]>>>(p1_start);
        cudaEventRecord(ev_1, stream[i]);
        cal_phase2_row<<<Rounds, threads, 0, stream[i]>>>(p1_start, i);
        cudaEventRecord(ev_2, stream[i]);

        for(int j=0; j<first_round; j++) {
            if(i==j) continue;


            p2_sub = p1_start + line_d * (j - i);
            //printf("\tp1=(%d,%d), p2=(%d,%d) stream %d\n", ROW_COL(p1_start), ROW_COL(p2_sub), j);
            cudaStreamWaitEvent(stream[j], ev_1, 0);
            cal_phase2_blk<<< 1, threads, 0, stream[j]>>>(p1_start, p2_sub);
            cudaStreamWaitEvent(stream[j], ev_2, 0);
            cal_phase3<<<Rounds, threads, 0, stream[j]>>>(p1_start, p2_sub, i);
        }
    }

    for(int i=first_round; i<Rounds; i++) {
        id[i] = (i / 2) % max_streams;
        s = id[i];

        ptr_h += line_h;
        ptr_d += line_d;
        cudaMemcpy2DAsync(ptr_d, pitch_bytes, ptr_h, b_rounds_bytes, b_rounds_bytes, block_size, cudaMemcpyHostToDevice, stream_m);
        cudaEventRecord(ev_m, stream_m);
        cudaStreamWaitEvent(stream[s], ev_m, 0);

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
            cudaEventRecord(ev_1, stream[s]);
            cudaStreamWaitEvent(stream_s, ev_1, 0);
        }
    }
    for(int i=1; i<Rounds; i+=2) {
        cudaEventRecord(ev_1, stream[id[i]]);
        cudaStreamWaitEvent(stream[id[i-1]], ev_1, 0);
    }


    //printf("R %d\n", Rounds);
    for (int r=first_round; r<Rounds; ++r) {

        cudaStreamWaitEvent(stream[id[r-1]], ev_2, 0);
        cal_phase1<<<1, threads, 0, stream_s>>>(p1_start);
        cudaEventRecord(ev_1, stream_s);
        cal_phase2_row<<<Rounds, threads, 0, stream_s>>>(p1_start, r);
        cudaEventRecord(ev_2, stream_s);

        if(r==Rounds-1) {
            ptr_h = Dist_h + r * line_n;
            ptr_d = Dist_d + r * line_d;
            cudaStreamWaitEvent(stream_m, ev_2, 0);
            cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, block_size, cudaMemcpyDeviceToHost, stream_m);
        }

        //printf("r %d\n", r);
        flag = 1;
        for(int i = (r+1) % Rounds; i != r; i = (i==Rounds-1) ? 0 : i+1) {
            s = id[i];
            p2_start = p1_start + line_d * (i-r);
            if(flag>0) {
                if(i==Rounds-1 || i==r-1 || i%2==1) {
                    //printf("\ti %d\n", i);
                    cudaStreamWaitEvent(stream[s], ev_1, 0);
                    cal_phase2_blk<<< 1, threads, 0, stream[s]>>>(p1_start, p2_start);
                    cudaStreamWaitEvent(stream[s], ev_2, 0);
                    cal_phase3<<<Rounds, threads, 0, stream[s]>>>(p1_start, p2_start, r);
                } else {

                    //printf("\ti %d %d\n", i, (i+1)%Rounds);
                    cudaStreamWaitEvent(stream[s], ev_1, 0);
                    cal_phase2_blk<<< 2, threads, 0, stream[s]>>>(p1_start, p2_start);
                    cudaStreamWaitEvent(stream[s], ev_2, 0);
                    cal_phase3_2<<<Rounds, threads, 0, stream[s]>>>(p1_start, p2_start, r);
                    flag--;
                }

                if(r==Rounds-1) {
                    cudaEventRecord(ev_1, stream[s]);
                    cudaStreamWaitEvent(stream_m, ev_1, 0);

                    ptr_d = Dist_d + i * line_d;
                    ptr_h = Dist_h + i * line_n;
                    cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, block_size, cudaMemcpyDeviceToHost, stream_m);
                    cnt++;
                    if(flag<=0) {
                        ptr_d += line_d;
                        ptr_h += line_n;
                        cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, block_size, cudaMemcpyDeviceToHost, stream_m);
                        cnt++;
                    }
                    if(cnt>=file_step) {
                        if(step_size>0) {
                            //printf("%d\n", i);
                            total -= step_size;
                            cudaEventSynchronize(ev_m);
                            fwrite(ptr_f, sizeof(int), step_size, fp);
                            ptr_f += step_size;
                        }
                        step_size = write_size * line_n;
                        cnt = 0;
                        cudaEventRecord(ev_m, stream_m);
                    }
                }
            } else {
                flag++;
            }

            if(i==r+1) {
                cudaEventRecord(ev_1, stream[s]);
                cudaStreamWaitEvent(stream_s, ev_1, 0);
            }
        }
        p1_start += diag_size;
    }
    //printf("cnt %d, sum %d, n %d\n", cnt, sum + cnt*block_size, n);
    if(step_size>0) {
        //printf("%d\n", i);
        total -= step_size;
        cudaEventSynchronize(ev_m);
        fwrite(ptr_f, sizeof(int), step_size, fp);
        ptr_f += step_size;
    }

    cudaEventRecord(ev_m, stream_m);
    cudaEventSynchronize(ev_m);
    fwrite(ptr_f, sizeof(int), total, fp);
    fclose(fp);

    for(int i=0; i<max_streams; i++) {
        cudaStreamDestroy(stream[i]);
    }
    cudaStreamDestroy(stream_s);
    cudaEventDestroy(ev_1);
    cudaStreamDestroy(stream_m);
    cudaEventDestroy(ev_2);
    cudaEventDestroy(ev_m);

    cudaFree(Dist_d);
    cudaFreeHost(Dist_h);

    cudaDeviceSynchronize();
}


void cuda_init()
{
    cudaStreamCreate(&stream_m);
    cudaMemcpy2DAsync(Dist_d, pitch_bytes, Dist_h, b_rounds_bytes, b_rounds_bytes, block_size, cudaMemcpyHostToDevice);

    line_h = block_size * b_rounds;
    line_d = block_size * pitch;
    line_n = block_size * n;
    diag_size = (pitch + 1) * block_size;

    cudaStreamCreate(&stream[0]);
    cal_phase1<<<1, threads, 0, stream[0]>>>(0);
    cudaEventCreateWithFlags(&ev_1, cudaEventDisableTiming);
    cudaEventRecord(ev_1, stream[0]);

    cal_phase2_row<<<Rounds, threads, 0, stream[0]>>>(0, 0);
    cudaEventCreateWithFlags(&ev_2, cudaEventDisableTiming);
    cudaEventRecord(ev_2, stream[0]);
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

    if(n>=10000) {
        max_streams = 6;
        first_round = 6;
    }

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

void block_FW_S(char *outFileName)
{
    FILE *outfile = fopen(outFileName, "w");
    int p1_start = 0, p2_start = 0;
    unsigned int *ptr_h = Dist_h, *ptr_d = Dist_d;
    int p2_sub;

    //printf("round 1\n");
    for(int i=1; i<Rounds; i++) {
        ptr_h += line_h;
        ptr_d += line_d;
        cudaMemcpy2DAsync(ptr_d, pitch_bytes, ptr_h, b_rounds_bytes, b_rounds_bytes, block_size, cudaMemcpyHostToDevice, stream_m);
        cudaEventCreateWithFlags(&ev_m, cudaEventDisableTiming);
        cudaEventRecord(ev_m, stream_m);

        cudaStreamCreate(&stream[i]);
        cudaStreamWaitEvent(stream[i], ev_m, 0);

        p2_start += line_d;
        //printf("\tp1=(%d,%d), p2=(%d,%d) stream %d\n", ROW_COL(p1_start), ROW_COL(p2_start), i);
        cudaStreamWaitEvent(stream[i], ev_1, 0);
        cal_phase2_blk<<< 1, threads, 0, stream[i]>>>(p1_start, p2_start);
        cudaStreamWaitEvent(stream[i], ev_2, 0);
        cal_phase3<<<Rounds, threads, 0, stream[i]>>>(p1_start, p2_start, 0);
    }
    for(int i=1; i<Rounds; i++) {
        p1_start += diag_size;
        //printf("round %d: p1=(%d,%d) stream %d\n", i, ROW_COL(p1_start), i);
        cal_phase1<<<1, threads, 0, stream[i]>>>(p1_start);
        cudaEventRecord(ev_1, stream[i]);
        cal_phase2_row<<<Rounds, threads, 0, stream[i]>>>(p1_start, i);
        cudaEventRecord(ev_2, stream[i]);

        for(int j=0; j<Rounds; j++) {
            if(i==j) continue;

            p2_sub = p1_start + line_d * (j - i);
            //printf("\tp1=(%d,%d), p2=(%d,%d) stream %d\n", ROW_COL(p1_start), ROW_COL(p2_sub), j);
            cudaStreamWaitEvent(stream[j], ev_1, 0);
            cal_phase2_blk<<< 1, threads, 0, stream[j]>>>(p1_start, p2_sub);
            cudaStreamWaitEvent(stream[j], ev_2, 0);
            cal_phase3<<<Rounds, threads, 0, stream[j]>>>(p1_start, p2_sub, i);
        }
    }

    ptr_h = Dist_h;
    ptr_d = Dist_d;
    for(int i=0; i<Rounds; i++) {
        cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, block_size, cudaMemcpyDeviceToHost, stream[i]);
        ptr_h += line_n;
        ptr_d += line_d;
    }
    cudaDeviceSynchronize();
    fwrite(Dist_h, sizeof(int), n*n, outfile);

    cudaStreamDestroy(stream_m);
    for(int i=0; i<max_streams; i++) {
        cudaStreamDestroy(stream[i]);
    }
    cudaEventDestroy(ev_1);
    cudaEventDestroy(ev_2);
    cudaEventDestroy(ev_m);

    fclose(outfile);

    cudaFree(Dist_d);
    cudaFreeHost(Dist_h);
}
