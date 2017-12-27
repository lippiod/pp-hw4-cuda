#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <omp.h>

#define CEIL(a, b) ( ((a) + (b) - 1) / (b) )
#define MIN(a, b) ( (a) < (b) ? (a) : (b) )
#define CAL_TIME ( 1e-6 * (temp_time.tv_usec - start_time.tv_usec) + (temp_time.tv_sec - start_time.tv_sec) )
#define C2I(i) ( ptr[i] - '0')
#define ROW_COL(__i) ( __i / line_d ), ( ( __i % pitch ) / block_size )

//const int tpb = 1024;
const int INF = 1000000000;
const int V = 20010;
const int block_size = 32;
//const int file_step = 10;
//const int write_size = 8;
const int max_devices = 2;
int max_streams = 4;
int first_round = 4;
dim3 threads(block_size, block_size);

void input(char *inFileName);
void block_FW(int tid);
void block_FW_S(int tid);
void split_strings(char *ptr);
void cuda_init(int tid);

__constant__ unsigned int *Dist;
__constant__ int pitch_d;


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

int n, n_bytes, out_size; // Number of vertices, edges
int Rounds, b_rounds, b_rounds_bytes;
int dist_size, dist_size_bytes;
int line_n;
unsigned int *Dist_h;
char buf[INF];
char *fstr;
FILE *outfile;
int last_line;

struct timeval start_time, temp_time;

unsigned int *Dist_d[max_devices];
int pitch_bytes, pitch;
int diag_size, line_d_bytes, line_d;

cudaStream_t stream[max_devices][8], stream_s[max_devices], stream_m[max_devices];
cudaEvent_t ev_1[max_devices], ev_2[max_devices], ev_m[max_devices];

int main(int argc, char* argv[])
{
    assert(argc==4);
    //block_size = atoi(argv[3]);

    //gettimeofday(&start_time, NULL);

    outfile = fopen(argv[2], "w+");
    input(argv[1]);
    //gettimeofday(&temp_time, NULL);
    //printf("input> %g s\n", CAL_TIME);

#pragma omp parallel default(shared) num_threads(2)
    {
        int tid = omp_get_thread_num();
        cudaSetDevice(tid);

        if(Rounds<=first_round) {
            block_FW_S(tid);
        } else {
            block_FW(tid);
            //printf("NOP\n");
        }
    }

    //gettimeofday(&temp_time, NULL);
    //printf("block_FW> %g s\n", CAL_TIME);

    return 0;
}


void cuda_init(int tid)
{
    cudaDeviceEnablePeerAccess(tid^1, 0);
    cudaError_t e = cudaMemcpy2DAsync(Dist_d[tid], pitch_bytes, Dist_h, n_bytes, n_bytes, block_size, cudaMemcpyHostToDevice);
    //fprintf(stderr, "memcpy -0 %s\n", cudaGetErrorString(e));

    cudaStreamCreate(&stream_m[tid]);
    cudaStreamCreate(&stream[tid][0]);
    cal_phase1<<<1, threads, 0, stream[tid][0]>>>(0);
    cudaEventCreateWithFlags(&ev_1[tid], cudaEventDisableTiming);
    cudaEventRecord(ev_1[tid], stream[tid][0]);

    cal_phase2_row<<<Rounds, threads, 0, stream[tid][0]>>>(0, 0);
    cudaEventCreateWithFlags(&ev_2[tid], cudaEventDisableTiming);
    cudaEventRecord(ev_2[tid], stream[tid][0]);

    cudaEventCreateWithFlags(&ev_m[tid], cudaEventDisableTiming);
}

void block_FW(int tid)
{
    int id_0[V], id_1[V];
    int p1_start = 0, p2_start = 0, p2_sub;
    unsigned int *ptr_h = Dist_h, *ptr_d = Dist_d[tid];
    //unsigned int step_size = 0;
    //int cnt = 0, total = n * n;
    int flag, bline = block_size;
    int oid = tid ^ 1, offset;
    cudaStream_t *sp, s;

    cuda_init(tid);

    id_0[0] = id_1[0] = 0;
    //printf("Round 1: row < first_round (in pivot)\n");
    for(int i=1; i<first_round; i++) {
        sp = &stream[tid][i];
        id_0[i] = (i%4) / max_devices;
        id_1[i] = i / 4;
        ptr_h += line_n;
        ptr_d += line_d;
        cudaMemcpy2DAsync(ptr_d, pitch_bytes, ptr_h, n_bytes, n_bytes, block_size, cudaMemcpyHostToDevice, stream_m[tid]);
        cudaEventRecord(ev_m[tid], stream_m[tid]);

        cudaStreamCreate(sp);
        s = *sp;
        cudaStreamWaitEvent(s, ev_1[tid], 0);
        cudaStreamWaitEvent(s, ev_m[tid], 0);

        p2_start += line_d;
        //printf("\tp1=(%d,%d), p2=(%d,%d) stream %d\n", ROW_COL(p1_start), ROW_COL(p2_start), i);
        cal_phase2_blk<<< 1, threads, 0, s>>>(p1_start, p2_start);
        cudaStreamWaitEvent(s, ev_2[tid], 0);
        cal_phase3<<<Rounds, threads, 0, s>>>(p1_start, p2_start, 0);
    }

    //printf("Round (2-first_round): row < first_round\n");
    for(int i=1; i<first_round; i++) {
        s = stream[tid][i];
        p1_start += diag_size;
        //printf("round %d: p1=(%d,%d) stream %d\n", i, ROW_COL(p1_start), i);
        cal_phase1<<<1, threads, 0, s>>>(p1_start);
        cudaEventRecord(ev_1[tid], s);
        cal_phase2_row<<<Rounds, threads, 0, s>>>(p1_start, i);
        cudaEventRecord(ev_2[tid], s);

        for(int j=0; j<first_round; j++) {
            if(i==j) continue;
            cudaStream_t sj = stream[tid][j];

            p2_sub = p1_start + line_d * (j - i);
            //printf("\tp1=(%d,%d), p2=(%d,%d) stream %d\n", ROW_COL(p1_start), ROW_COL(p2_sub), j);
            cudaStreamWaitEvent(sj, ev_1[tid], 0);
            cal_phase2_blk<<< 1, threads, 0, sj>>>(p1_start, p2_sub);
            cudaStreamWaitEvent(sj, ev_2[tid], 0);
            cal_phase3<<<Rounds, threads, 0, sj>>>(p1_start, p2_sub, i);
        }
    }

    for(int i=0; i<max_streams; i++) {
        cudaEventRecord(ev_1[tid], stream[tid][i]);
        for(int j=0; j<max_streams; j++) {
            if(i==j) continue;
            cudaStreamWaitEvent(stream[tid][j], ev_1[tid], 0);
        }
    }

    //printf("Round (1-first_round): other rows\n");
    for(int i=first_round; i<Rounds; i++) {
        id_0[i] = (i%4) / max_devices;
        id_1[i] = (i/4) % max_streams;
        s = stream[tid][id_1[i]];

        ptr_h += line_n;
        ptr_d += line_d;
        p2_start += line_d;
        if(i==first_round || id_0[i]==tid) {
            bline = i==Rounds-1 ? last_line : block_size;
            cudaMemcpy2DAsync(ptr_d, pitch_bytes, ptr_h, n_bytes, n_bytes, bline, cudaMemcpyHostToDevice, stream_m[tid]);
            cudaEventRecord(ev_m[tid], stream_m[tid]);
            cudaStreamWaitEvent(s, ev_m[tid], 0);

            p1_start = 0;
            p2_sub = p2_start;
            //printf("\nround %d seq\n", i);
            for(int r=0; r<first_round; r++) {
                //printf("\tp1=(%d,%d), p2=(%d,%d) stream %d\n", ROW_COL(p1_start), ROW_COL(p2_sub), s);

                cal_phase2_blk<<< 1, threads, 0, s>>>(p1_start, p2_sub);
                cal_phase3<<<Rounds, threads, 0, s>>>(p1_start, p2_sub, r);
                p1_start += diag_size;
                p2_sub += block_size;
            }
            if(i==first_round) {
                cudaStreamCreate(&stream_s[tid]);
                cudaEventRecord(ev_1[tid], s);
                cudaStreamWaitEvent(stream_s[tid], ev_1[tid], 0);
            }
        }
    }

    //printf("R %d\n", Rounds);
    for (int r=first_round; r<Rounds; ++r) {
#pragma omp barrier

        if(id_0[r]==oid)
            cudaStreamWaitEvent(stream_s[tid], ev_m[oid], 0);
        //cudaStreamWaitEvent(stream[tid][id_1[r-1]], ev_2[tid], 0);
        cal_phase1<<<1, threads, 0, stream_s[tid]>>>(p1_start);
        cudaEventRecord(ev_1[tid], stream_s[tid]);

        cal_phase2_row<<<Rounds, threads, 0, stream_s[tid]>>>(p1_start, r);
        cudaEventRecord(ev_2[tid], stream_s[tid]);

        if(r==Rounds-1 && tid==1)
        {
            ptr_h = Dist_h + r * line_n;
            ptr_d = Dist_d[tid] + r * line_d;
            cudaStreamWaitEvent(stream_m[tid], ev_2[tid], 0);
            cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, n - r*block_size, cudaMemcpyDeviceToHost, stream_m[tid]);
            //cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, last_line, cudaMemcpyDeviceToHost);
        }

        //printf("r %d\n", r);
        flag = 1;
        for(int i = (r+1) % Rounds; i != r; i = (i==Rounds-1) ? 0 : i+1) {
            if(id_0[i]==oid) continue;

            s = stream[tid][id_1[i]];
            p2_start = p1_start + line_d * (i-r);
            if(flag>0) {
                if(i+1==Rounds || i+1==r || id_0[i+1]==oid) {
                    //printf("\ti %d\n", i);
                    cudaStreamWaitEvent(s, ev_1[tid], 0);
                    cal_phase2_blk<<< 1, threads, 0, s>>>(p1_start, p2_start);
                    cudaStreamWaitEvent(s, ev_2[tid], 0);
                    cal_phase3<<<Rounds, threads, 0, s>>>(p1_start, p2_start, r);
                } else {
                    //printf("\ti %d %d\n", i, (i+1)%Rounds);
                    cudaStreamWaitEvent(s, ev_1[tid], 0);
                    cal_phase2_blk<<< 2, threads, 0, s>>>(p1_start, p2_start);
                    cudaStreamWaitEvent(s, ev_2[tid], 0);
                    cal_phase3_2<<<Rounds, threads, 0, s>>>(p1_start, p2_start, r);
                    flag--;
                }

                if(r==Rounds-1) {

                    cudaEventRecord(ev_1[tid], s);
                    cudaStreamWaitEvent(stream_m[tid], ev_1[tid], 0);

                    bline = flag==0 ? block_size * 2 : block_size;
                    ptr_d = Dist_d[tid] + i * line_d;
                    ptr_h = Dist_h + i * line_n;
                    cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, bline, cudaMemcpyDeviceToHost, stream_m[tid]);

                }
            } else {
                flag++;
            }

            if(i==r+1) {
                cudaEventRecord(ev_1[tid], s);
                cudaStreamWaitEvent(stream_s[tid], ev_1[tid], 0);
                cudaStreamWaitEvent(stream_m[tid], ev_1[tid], 0);
                offset = line_d * i;
                cudaMemcpyAsync(Dist_d[oid]+offset, Dist_d[tid]+offset, line_d_bytes, cudaMemcpyDeviceToDevice, stream_m[tid]);
                cudaEventRecord(ev_m[tid], stream_m[tid]);
            }
        }
        p1_start += diag_size;
    }

#pragma omp sections
    {
#pragma omp section
        {
            //fprintf(stderr, "%d %d fwrite %d remain\n", omp_get_thread_num(), ptr_f - Dist_h, total);
            for(int d=0; d<max_devices; d++) {
                cudaSetDevice(d);
                cudaDeviceSynchronize();
            }
            msync(Dist_h, out_size, MS_SYNC);
            munmap(Dist_h, out_size);
            fclose(outfile);
        }
#pragma omp section
        {
            //fprintf(stderr, "%d %d cuda\n", omp_get_thread_num(), ptr_f - Dist_h);
            for(int d=0; d<max_devices; d++) {
                cudaSetDevice(d);
                cudaDeviceSynchronize();
                for(int i=0; i<max_streams; i++) {
                    cudaStreamDestroy(stream[d][i]);
                }
                cudaStreamDestroy(stream_s[d]);
                cudaStreamDestroy(stream_m[d]);
                cudaEventDestroy(ev_1[d]);
                cudaEventDestroy(ev_2[d]);
                cudaEventDestroy(ev_m[d]);

                cudaFree(Dist_d[d]);
            }
        }
    }
}


size_t m, sz, fsize;
void input(char *inFileName)
{
    char *tok, *next_tok;
    size_t p_bytes[max_devices];
    FILE *infile = fopen(inFileName, "rb");
    char temp[30];

    fseek(infile, 0L, SEEK_END);
    sz = ftell(infile);
    fseek(infile, 0L, SEEK_SET);

    fgets(temp, 20, infile);
    tok = strtok_r(temp, " ", &next_tok);
    n = atoi(tok);
    tok = strtok_r(NULL, "\n", &next_tok);
    m = atoi(tok);

    Rounds = CEIL(n, block_size);
    b_rounds = block_size * Rounds;
    b_rounds_bytes = b_rounds * sizeof(int);

    //gettimeofday(&temp_time, NULL);
    //printf("before parsing> %g s\n", CAL_TIME);
#pragma omp parallel default(shared) private(temp_time) num_threads(2)
    {
#pragma omp sections
        {
#pragma omp section
            {
                int fd = fileno(outfile);
                fstr = sz<INF ? buf : (char *) malloc(sz+10);
                fsize = fread(fstr, sizeof(char), sz+10, infile);
                fclose(infile);
                fstr[fsize] = '\0';

                //cudaMallocHost(&Dist_h, dist_size_bytes);
                n_bytes = n * sizeof(int);
                last_line = n - (Rounds-1) * block_size;
                out_size = n * n_bytes;
                ftruncate(fd, out_size);

                //gettimeofday(&temp_time, NULL);
                //printf("\tmmap 0> %g s\n", CAL_TIME);

                Dist_h = (unsigned int *) mmap(NULL, out_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_POPULATE, fd, 0);
                if(Dist_h==MAP_FAILED) {
                    fprintf(stderr, "mmap faild 1\n");
                    exit(1);
                }
                //gettimeofday(&temp_time, NULL);
                //printf("\tmmap 1> %g s\n", CAL_TIME);
/*
                Dist_h = (unsigned int *) mmap(Dist_h, out_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_POPULATE|MAP_FIXED, fd, 0);
                if(Dist_h==MAP_FAILED) {
                    fprintf(stderr, "mmap faild 2\n");
                    exit(1);
                }
*/
                //gettimeofday(&temp_time, NULL);
                //printf("\tmmap 2> %g s\n", CAL_TIME);
                //fprintf(stderr, "mmap success %d\n", Dist_h);

                //cudaError_t e = cudaHostRegister(Dist_h, out_size, cudaHostRegisterPortable);
                //fprintf(stderr, "HostReg %s\n", cudaGetErrorString(e));
                memset(Dist_h, 64, out_size);
                for (int i = 0; i < n*n; i+=n+1) {
                    Dist_h[i] = 0;
                }
                //fprintf(stderr, "memset success\n");

                //gettimeofday(&temp_time, NULL);
                //printf("\tfile read done> %g s\n", CAL_TIME);

                split_strings(fstr);
                if(sz>=INF)
                    free(fstr);

                //gettimeofday(&temp_time, NULL);
                //printf("\tparsing done> %g s\n", CAL_TIME);
            }
#pragma omp section
            {
                line_n = block_size * n;
                if(n>=10000) {
                    max_streams = 6;
                    first_round = 6;
                }
                for(int d=0; d<max_devices; d++) {
                    int dev;
                    cudaGetDeviceCount(&dev);

                    cudaSetDevice(d);
                    cudaMallocPitch(&Dist_d[d], &p_bytes[d], b_rounds_bytes, b_rounds);
                    pitch_bytes = p_bytes[d];
                    pitch = pitch_bytes / sizeof(int);
                    line_d = block_size * pitch;
                    line_d_bytes = line_d * sizeof(int);
                    diag_size = (pitch + 1) * block_size;
                    cudaMemcpyToSymbolAsync(Dist, &Dist_d[d], sizeof(Dist_d[d]), 0);
                    cudaMemcpyToSymbolAsync(pitch_d, &pitch, sizeof(pitch), 0);
                    cudaMemset2DAsync(Dist_d[d], pitch_bytes, 64, b_rounds_bytes, b_rounds);
                }

                //gettimeofday(&temp_time, NULL);
                //printf("\tcuda allocate done> %g s\n", CAL_TIME);
            }
        }
    }
}

void split_strings(char *ptr)
{
    int a, b, v;
    while(m-->0) {
        if(ptr[1]==' ') {
            a = C2I(0);
            ptr += 2;
        } else if(ptr[2]==' ') {
            a = C2I(0) * 10 + C2I(1);
            ptr += 3;
        } else if(ptr[3]==' ') {
            a = C2I(0) * 100 + C2I(1) * 10 + C2I(2);
            ptr += 4;
        } else if(ptr[4]==' ') {
            a = C2I(0) * 1000 + C2I(1) * 100 + C2I(2) * 10 + C2I(3);
            ptr += 5;
        } else {
            a = C2I(0) * 10000 + C2I(1) * 1000 + C2I(2) * 100 + C2I(3) * 10 + C2I(4);
            ptr += 6;
        }

        if(ptr[1]==' ') {
            b = C2I(0);
            ptr += 2;
        } else if(ptr[2]==' ') {
            b = C2I(0) * 10 + C2I(1);
            ptr += 3;
        } else if(ptr[3]==' ') {
            b = C2I(0) * 100 + C2I(1) * 10 + C2I(2);
            ptr += 4;
        } else if(ptr[4]==' ') {
            b = C2I(0) * 1000 + C2I(1) * 100 + C2I(2) * 10 + C2I(3);
            ptr += 5;
        } else {
            b = C2I(0) * 10000 + C2I(1) * 1000 + C2I(2) * 100 + C2I(3) * 10 + C2I(4);
            ptr += 6;
        }

        if(ptr[1]=='\n') {
            v = C2I(0);
            ptr += 2;
        } else if(ptr[2]=='\n') {
            v = C2I(0) * 10 + C2I(1);
            ptr += 3;
        } else {
            v = C2I(0) * 100 + C2I(1) * 10 + C2I(2);
            ptr += 4;
        }

        Dist_h[ n * a + b ] = v;
    }
}

void block_FW_S(int tid)
{
    int p1_start = 0, p2_start = 0, offset;
    unsigned int *ptr_h = Dist_h, *ptr_d = Dist_d[tid];
    int p2_sub, bline;

    cuda_init(tid);

    //printf("round 1\n");
    for(int i=1; i<Rounds; i++) {
        ptr_h += line_n;
        ptr_d += line_d;
        bline = i==Rounds-1 ? n - i * block_size : block_size;
        cudaMemcpy2DAsync(ptr_d, pitch_bytes, ptr_h, n_bytes, n_bytes, bline, cudaMemcpyHostToDevice, stream_m[tid]);
        cudaEventRecord(ev_m[tid], stream_m[tid]);

        cudaStreamCreate(&stream[tid][i]);
        cudaStreamWaitEvent(stream[tid][i], ev_m[tid], 0);

        p2_start += line_d;
        //printf("\tp1=(%d,%d), p2=(%d,%d) stream %d\n", ROW_COL(p1_start), ROW_COL(p2_start), i);
        cudaStreamWaitEvent(stream[tid][i], ev_1[tid], 0);
        cal_phase2_blk<<< 1, threads, 0, stream[tid][i]>>>(p1_start, p2_start);
        cudaStreamWaitEvent(stream[tid][i], ev_2[tid], 0);
        cal_phase3<<<Rounds, threads, 0, stream[tid][i]>>>(p1_start, p2_start, 0);
    }
    //fprintf(stderr, "%d first round done\n", tid);

#pragma omp barrier

    for(int i=1; i<Rounds; i++) {
        p1_start += diag_size;
        //printf("round %d: p1=(%d,%d) stream %d\n", i, ROW_COL(p1_start), i);
        cal_phase1<<<1, threads, 0, stream[tid][i]>>>(p1_start);
        cudaEventRecord(ev_1[tid], stream[tid][i]);
        cal_phase2_row<<<Rounds, threads, 0, stream[tid][i]>>>(p1_start, i);
        cudaEventRecord(ev_2[tid], stream[tid][i]);

        for(int j=0; j<Rounds; j++) {
            if(i==j) continue;
            int dev = j % 2;

            if(dev==tid) {
                p2_sub = p1_start + line_d * (j - i);
                //printf("\tp1=(%d,%d), p2=(%d,%d) stream %d\n", ROW_COL(p1_start), ROW_COL(p2_sub), j);
                cudaStreamWaitEvent(stream[tid][j], ev_1[tid], 0);
                cal_phase2_blk<<< 1, threads, 0, stream[tid][j]>>>(p1_start, p2_sub);
                cudaStreamWaitEvent(stream[tid][j], ev_2[tid], 0);
                cal_phase3<<<Rounds, threads, 0, stream[tid][j]>>>(p1_start, p2_sub, i);
            }

#pragma omp barrier
            if(j==i+1)
#pragma omp single
            {
                offset = j * line_d;
                cudaMemcpy(Dist_d[dev^1]+offset, Dist_d[dev]+offset, line_d_bytes, cudaMemcpyDeviceToDevice);
                //cudaError_t e = cudaMemcpyPeer(Dist_d[dev^1]+offset, dev^1, Dist_d[dev]+offset, dev, line_d_bytes);
            }
        }
    }
    //fprintf(stderr, "%d all rounds done\n", tid);

    for(int i=tid; i<Rounds; i+=2) {
        ptr_h = Dist_h + i * line_n;
        ptr_d = Dist_d[tid] + i * line_d;
        bline = i==Rounds-1 ? last_line : block_size;
        cudaError_t e = cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, bline, cudaMemcpyDeviceToHost, stream[tid][i]);
        //fprintf(stderr, "memcpy %d %s\n", i, cudaGetErrorString(e));
    }
    if(Rounds==1)
        cudaMemcpy2DAsync(Dist_h, n_bytes, Dist_d[tid], pitch_bytes, n_bytes, block_size, cudaMemcpyDeviceToHost);

    for(int d=0; d<max_devices; d++) {
        cudaSetDevice(d);
        cudaDeviceSynchronize();
    }

    gettimeofday(&temp_time, NULL);
    //printf("before output> %g s\n", CAL_TIME);
#pragma omp sections
    {
#pragma omp section
        {
            //fprintf(stderr, "%d - write file\n", tid);
            msync(Dist_h, out_size, MS_SYNC);
            munmap(Dist_h, out_size);
            fclose(outfile);

            //gettimeofday(&temp_time, NULL);
            //printf("\twrite done> %g s\n", CAL_TIME);
        }
#pragma omp section
        {
            //fprintf(stderr, "%d - cleanup\n", tid);
            for(int d=0; d<max_devices; d++) {
                cudaSetDevice(d);
                cudaStreamDestroy(stream_m[d]);
                for(int i=0; i<Rounds; i++) {
                    cudaStreamDestroy(stream[d][i]);
                }
                cudaEventDestroy(ev_1[d]);
                cudaEventDestroy(ev_2[d]);
                cudaEventDestroy(ev_m[d]);
                cudaFree(Dist_d[d]);
            }

            //gettimeofday(&temp_time, NULL);
            //printf("\tcuda free done> %g s\n", CAL_TIME);
        }
    }
}
