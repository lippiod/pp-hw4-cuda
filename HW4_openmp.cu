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
int max_row = 4;
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
    __shared__ unsigned int block_dist[block_size][block_size+1];
    __shared__ unsigned int pivot_dist[block_size][block_size+1];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * pitch_d + tx;
    int column = block_size * (blockIdx.x - r);
    int block_index, pivot_index;
    unsigned int origin, blk_dist, new_dist;

    pivot_index = pivot + tid;

    if(blockIdx.x==r) // pivot block
        return;
/*
    block_index = pivot_index + column;
    block_dist[ty][tx] = origin = Dist[block_index];
    __syncthreads();

    pivot_dist[ty][tx] = Dist[pivot_index];

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
*/
    pivot_dist[ty][tx] = Dist[pivot_index];
    block_index = pivot_index + column;
    block_dist[tx][ty] = origin = Dist[block_index];
    __syncthreads();

    if(origin > INF)
        Dist[block_index] = origin = INF;

    blk_dist = block_dist[ty][tx];
    for(int k=0; k<block_size; k++) {
        new_dist = pivot_dist[tx][k] + block_dist[ty][k];

        if (blk_dist > new_dist)
            block_dist[ty][tx] = blk_dist = new_dist;
    }
    __syncthreads();

    blk_dist = block_dist[tx][ty];
    if(origin > blk_dist)
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
    __syncthreads();

    block_index = p2_pivot + tid + blockIdx.x * pitch_d * block_size;
    block_dist[ty][tx] = origin = Dist[block_index];

    blk_dist = origin;
    for(int k=0; k<block_size-1; k++) {
        new_dist = block_dist[ty][k] + pivot_dist[k][tx];

        //if (block_dist[ty][tx] > new_dist)
            //block_dist[ty][tx] = new_dist;
        if(blk_dist > new_dist)
            block_dist[ty][tx] = blk_dist = new_dist;
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

__global__ void cal_phase3_n(int p1_pivot, int p2_pivot, int r, int n)
{
    __shared__ unsigned int pvR_dist[block_size][block_size];
    __shared__ unsigned int pvC_dist[2][block_size][block_size];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * pitch_d + tx;
    int col_diff = (blockIdx.x - r) * block_size;
    int row_diff = pitch_d * block_size;
    int b1_index, b2_index, p_index;
    unsigned int origin, b1_dist, b2_dist;
    unsigned int inter[block_size], new1, new2;
    int p1 = 0, p2 = 1;

    p_index = p1_pivot + tid + col_diff;

    if(col_diff==0) // pivots
        return;

    pvR_dist[ty][tx] = Dist[p_index];
    __syncthreads();
    for(int k=0; k<block_size; k++)
        inter[k] = pvR_dist[k][tx];

    p_index = p2_pivot + tid;
    pvC_dist[p1][ty][tx] = Dist[p_index];
    b1_index = p_index + col_diff;
    b1_dist = origin = Dist[b1_index];
    while(n-->1) {
        p_index += row_diff;
        pvC_dist[p2][ty][tx] = Dist[p_index];
        b2_index = b1_index + row_diff;
        b2_dist = Dist[b2_index];

        for(int k=0; k<block_size; k+=2) {
            new1 = pvC_dist[p1][ty][k] + inter[k];
            new2 = pvC_dist[p1][ty][k+1] + inter[k+1];
            if (b1_dist > new1)
                b1_dist = new1;
            if (b1_dist > new2)
                b1_dist = new2;
        }
        if (origin > b1_dist)
            Dist[b1_index] = b1_dist;
        //Dist[b1_index] = MIN(origin, b1_dist);

        p1 ^= 1;
        p2 ^= 1;
        b1_dist = origin = b2_dist;
        b1_index = b2_index;
    }
    for(int k=0; k<block_size; k+=2) {
        new1 = pvC_dist[p1][ty][k] + inter[k];
        new2 = pvC_dist[p1][ty][k+1] + inter[k+1];
        if (b1_dist > new1)
            b1_dist = new1;
        if (b1_dist > new2)
            b1_dist = new2;
    }
    if (origin > b1_dist)
        Dist[b1_index] = b1_dist;
    //Dist[b1_index] = MIN(origin, b1_dist);
}

int n, n_bytes, out_size; // Number of vertices, edges
int Rounds, b_rounds, b_rounds_bytes;
int dist_size, dist_size_bytes;
int line_n;
unsigned int *Dist_h;
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

    gettimeofday(&start_time, NULL);

    outfile = fopen(argv[2], "w+");
    input(argv[1]);
    gettimeofday(&temp_time, NULL);
    //fprintf(stderr, "input> %g s\n", CAL_TIME);

#pragma omp parallel default(shared) num_threads(2)
    {
        int tid = omp_get_thread_num();
        cudaSetDevice(tid);

        if(Rounds<=8) {
            block_FW_S(tid);
        } else {
            block_FW(tid);
            //printf("NOP\n");
        }
    }

    gettimeofday(&temp_time, NULL);
    //fprintf(stderr, "block_FW> %g s\n", CAL_TIME);

    return 0;
}


void cuda_init(int tid)
{
    cudaDeviceEnablePeerAccess(tid^1, 0);
    int bline = Rounds==1 ? n : block_size;
    cudaMemcpy2DAsync(Dist_d[tid], pitch_bytes, Dist_h, n_bytes, n_bytes, bline, cudaMemcpyHostToDevice);
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
    int row_dev = max_row * max_devices;
    int id_0[V], id_1[V], do_r[V], row;
    int p1_start = 0, p2_start = 0, p2_sub;
    unsigned int *ptr_h = Dist_h, *ptr_d = Dist_d[tid];
    //unsigned int step_size = 0;
    //int cnt = 0, total = n * n;
    int flag, bline = block_size;
    int oid = tid ^ 1, offset;
    cudaStream_t *sp, s;

    cuda_init(tid);

    id_0[0] = id_1[0] = 0;
    do_r[0] = max_row;
    //printf("Round 1: row < first_round (in pivot)\n");
    for(int i=1; i<first_round; i++) {
        sp = &stream[tid][i];
        id_0[i] = i % row_dev / max_row;
        id_1[i] = i / row_dev;
        do_r[i] = max_row - i % max_row;
        ptr_h += line_n;
        ptr_d += line_d;
        cudaMemcpy2DAsync(ptr_d, pitch_bytes, ptr_h, n_bytes, n_bytes, block_size, cudaMemcpyHostToDevice, stream_m[tid]);
        cudaEventRecord(ev_m[tid], stream_m[tid]);

        cudaStreamCreate(sp);
        //fprintf(stderr, "r %d t%d id%d s%d\n", i, tid, i, *sp);
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
    flag = 1;
    for(int i=first_round; i<Rounds; i++) {
        id_0[i] = i % row_dev / max_row;
        id_1[i] = i / row_dev % max_streams;
        do_r[i] = max_row - i % max_row;
        if(i + do_r[i] > Rounds)
            do_r[i] = Rounds - i;

        s = stream[tid][id_1[i]];
        //fprintf(stderr, "r %d t%d id%d s%d\n", i, tid, id_1[i], s);

        ptr_h += line_n;
        ptr_d += line_d;
        p2_start += line_d;
        if(i==first_round || id_0[i]==tid) {
            if(flag>0) {
                if(id_0[i]!=tid) {
                    row = 1;
                    bline = i==Rounds-1 ? last_line : block_size;
                } else {
                    row = do_r[i];
                    bline = i+do_r[i]==Rounds ? last_line + (do_r[i]-1) * block_size : do_r[i] * block_size;
                }
                cudaMemcpy2DAsync(ptr_d, pitch_bytes, ptr_h, n_bytes, n_bytes, bline, cudaMemcpyHostToDevice, stream_m[tid]);
                cudaEventRecord(ev_m[tid], stream_m[tid]);
                cudaStreamWaitEvent(s, ev_m[tid], 0);

                p1_start = 0;
                p2_sub = p2_start;
                //printf("\nround %d seq\n", i);
                for(int r=0; r<first_round; r++) {
                    //printf("\tp1=(%d,%d), p2=(%d,%d) stream %d\n", ROW_COL(p1_start), ROW_COL(p2_sub), s);

                    cal_phase2_blk<<<row, threads, 0, s>>>(p1_start, p2_sub);
                    cal_phase3_n<<<Rounds, threads, 0, s>>>(p1_start, p2_sub, r, row);
                    p1_start += diag_size;
                    p2_sub += block_size;
                }
                if(i==first_round) {
                    cudaStreamCreate(&stream_s[tid]);
                    cudaEventRecord(ev_1[tid], s);
                    cudaStreamWaitEvent(stream_s[tid], ev_1[tid], 0);
                }
                flag -= row - 1;
            } else {
                flag++;
            }
        }
    }

    //printf("R %d\n", Rounds);
    for (int r=first_round; r<Rounds; ++r) {
#pragma omp barrier
        if(id_0[r]==oid)
            cudaStreamWaitEvent(stream_s[tid], ev_m[oid], 0);
        cal_phase1<<<1, threads, 0, stream_s[tid]>>>(p1_start);
        cudaEventRecord(ev_1[tid], stream_s[tid]);
        cal_phase2_row<<<Rounds, threads, 0, stream_s[tid]>>>(p1_start, r);
        cudaEventRecord(ev_2[tid], stream_s[tid]);

        if(r==Rounds-1 && tid==1) {
            ptr_h = Dist_h + r * line_n;
            ptr_d = Dist_d[tid] + r * line_d;
            cudaStreamWaitEvent(stream_m[tid], ev_2[tid], 0);
            cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, n - r*block_size, cudaMemcpyDeviceToHost, stream_m[tid]);
            //cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, last_line, cudaMemcpyDeviceToHost);
        }

        //printf("r %d\n", r);
        if(r+1<Rounds && id_0[r+1]==tid) {
            s = stream[tid][id_1[r+1]];
            p2_start = p1_start + line_d;
            cudaStreamWaitEvent(s, ev_1[tid], 0);
            cal_phase2_blk<<<1, threads, 0, s>>>(p1_start, p2_start);
            cudaStreamWaitEvent(s, ev_2[tid], 0);
            cal_phase3<<<Rounds, threads, 0, s>>>(p1_start, p2_start, r);

            cudaEventRecord(ev_m[tid], s);
            cudaStreamWaitEvent(stream_s[tid], ev_m[tid], 0);
            cudaStreamWaitEvent(stream_m[tid], ev_m[tid], 0);
            offset = line_d * (r+1);
            cudaMemcpyAsync(Dist_d[oid]+offset, Dist_d[tid]+offset, line_d_bytes, cudaMemcpyDeviceToDevice, stream_m[tid]);
            cudaEventRecord(ev_m[tid], stream_m[tid]);
        }
        flag = 1;
        for(int i = (r+1) % Rounds; i != r; i = (i==Rounds-1) ? 0 : i+1) {
            if(id_0[i]==oid || i==r+1) continue;

            s = stream[tid][id_1[i]];
            p2_start = p1_start + line_d * (i-r);
            if(flag>0) {
                row = (i<r && i+do_r[i]>r) ? r - i : do_r[i];
                cudaStreamWaitEvent(s, ev_1[tid], 0);
                cal_phase2_blk<<<row, threads, 0, s>>>(p1_start, p2_start);
                cudaStreamWaitEvent(s, ev_2[tid], 0);
                cal_phase3_n<<<Rounds, threads, 0, s>>>(p1_start, p2_start, r, row);
                flag -= row - 1;
                
                if(r==Rounds-1) {
                    cudaEventRecord(ev_m[tid], s);
                    cudaStreamWaitEvent(stream_m[tid], ev_m[tid], 0);

                    bline = row * block_size;
                    ptr_d = Dist_d[tid] + i * line_d;
                    ptr_h = Dist_h + i * line_n;
                    cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, bline, cudaMemcpyDeviceToHost, stream_m[tid]);
                }
            } else {
                flag++;
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
            cudaSetDevice(0);
            cudaDeviceSynchronize();
            for(int i=0; i<max_streams; i++) {
                cudaStreamDestroy(stream[0][i]);
            }
            cudaStreamDestroy(stream_s[0]);
            cudaStreamDestroy(stream_m[0]);
            cudaEventDestroy(ev_1[0]);
            cudaEventDestroy(ev_2[0]);
            cudaEventDestroy(ev_m[0]);

            cudaFree(Dist_d[0]);
        }
#pragma omp section
        {
            //fprintf(stderr, "%d %d cuda\n", omp_get_thread_num(), ptr_f - Dist_h);
            cudaSetDevice(1);
            cudaDeviceSynchronize();
            for(int i=0; i<max_streams; i++) {
                cudaStreamDestroy(stream[1][i]);
            }
            cudaStreamDestroy(stream_s[1]);
            cudaStreamDestroy(stream_m[1]);
            cudaEventDestroy(ev_1[1]);
            cudaEventDestroy(ev_2[1]);
            cudaEventDestroy(ev_m[1]);

            cudaFree(Dist_d[1]);
        }
    }
}


size_t m, sz, fsize;
void input(char *inFileName)
{
    char *tok_1, *tok_2;
    size_t p_bytes[max_devices];
    FILE *infile = fopen(inFileName, "r");
    char temp[30];

    fseek(infile, 0L, SEEK_END);
    sz = ftell(infile);
    fseek(infile, 0L, SEEK_SET);

    fstr = (char *) mmap(NULL, sz, PROT_READ, MAP_PRIVATE|MAP_POPULATE, fileno(infile), 0);
    if(fstr==MAP_FAILED) {
        fprintf(stderr, "mmap faild fstr\n");
        exit(1);
    }
    tok_1 = strchr(fstr, ' ');
    strncpy(temp, fstr, tok_1-fstr);
    n = atoi(temp);
    tok_1++;
    tok_2 = strchr(tok_1, '\n');
    strncpy(temp, tok_1, tok_2-tok_1);
    m = atoi(temp);
    tok_2++;

    Rounds = CEIL(n, block_size);
    b_rounds = block_size * Rounds;
    b_rounds_bytes = b_rounds * sizeof(int);

    gettimeofday(&temp_time, NULL);
    fprintf(stderr, "before parsing> %g s\n", CAL_TIME);
#pragma omp parallel default(shared) private(temp_time)
    {
#pragma omp sections
        {
#pragma omp section
            {
                int fd = fileno(outfile);

                n_bytes = n * sizeof(int);
                out_size = n * n_bytes;
                ftruncate(fd, out_size);

                //gettimeofday(&temp_time, NULL);
                //printf("\tmmap 0> %g s\n", CAL_TIME);

                Dist_h = (unsigned int *) mmap(NULL, out_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
                if(Dist_h==MAP_FAILED) {
                    fprintf(stderr, "mmap faild Dist_h\n");
                    exit(1);
                }
                //gettimeofday(&temp_time, NULL);
                //printf("\tmmap 1> %g s\n", CAL_TIME);

                memset(Dist_h, 64, out_size);
                for (int i = 0; i < n*n; i+=n+1)
                    Dist_h[i] = 0;
                //gettimeofday(&temp_time, NULL);
                //printf("\tfile read done> %g s\n", CAL_TIME);

                split_strings(tok_2);
                munmap(fstr, sz);
                fclose(infile);

                gettimeofday(&temp_time, NULL);
                fprintf(stderr, "\tparsing done (%d)> %g s\n", omp_get_thread_num(), CAL_TIME);
            }
#pragma omp section
            {
                line_n = block_size * n;
                if(n>=10000) {
                    max_streams = 6;
                    first_round = 6;
                }
                max_row = (Rounds+max_streams*2-1) / (max_streams * 2);
                //max_row = Rounds;
                //fprintf(stderr, "n %d, Rounds %d, streams %d, rows %d\n", n, Rounds, max_streams, max_row);
                cudaSetDevice(0);
                cudaMallocPitch(&Dist_d[0], &p_bytes[0], b_rounds_bytes, b_rounds);
                pitch_bytes = p_bytes[0];
                pitch = pitch_bytes / sizeof(int);
                line_d = block_size * pitch;
                line_d_bytes = line_d * sizeof(int);
                diag_size = (pitch + 1) * block_size;
                cudaMemcpyToSymbolAsync(Dist, &Dist_d[0], sizeof(Dist_d[0]), 0);
                cudaMemcpyToSymbolAsync(pitch_d, &pitch, sizeof(pitch), 0);
                cudaMemset2DAsync(Dist_d[0], pitch_bytes, 64, b_rounds_bytes, b_rounds);

                gettimeofday(&temp_time, NULL);
                fprintf(stderr, "\tcuda allocate done (%d)> %g s\n", omp_get_thread_num(), CAL_TIME);
            }
#pragma omp section
            {
                last_line = n - (Rounds-1) * block_size;

                cudaSetDevice(1);
                cudaMallocPitch(&Dist_d[1], &p_bytes[1], b_rounds_bytes, b_rounds);
                pitch_bytes = p_bytes[1];
                pitch = pitch_bytes / sizeof(int);
                cudaMemcpyToSymbolAsync(Dist, &Dist_d[1], sizeof(Dist_d[1]), 0);
                cudaMemcpyToSymbolAsync(pitch_d, &pitch, sizeof(pitch), 0);
                cudaMemset2DAsync(Dist_d[1], pitch_bytes, 64, b_rounds_bytes, b_rounds);

                gettimeofday(&temp_time, NULL);
                fprintf(stderr, "\tcuda allocate done (%d)> %g s\n", omp_get_thread_num(), CAL_TIME);
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
    int p2_sub, bline, oid = tid ^ 1;

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

    for(int i=1; i<Rounds; i++) {
#pragma omp barrier
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
                if(j==i+1) {
                    offset = j * line_d;
                    cudaEventRecord(ev_m[tid], stream[tid][j]);
                    cudaStreamWaitEvent(stream_m[tid], ev_m[tid], 0);
                    cudaMemcpyAsync(Dist_d[oid]+offset, Dist_d[tid]+offset, line_d_bytes, cudaMemcpyDeviceToDevice, stream_m[tid]);
                    cudaEventRecord(ev_m[tid], stream_m[tid]);
                    cudaStreamWaitEvent(stream[oid][j], ev_m[tid], 0);
                }
            }
        }
    }
    //fprintf(stderr, "%d all rounds done\n", tid);

    if(Rounds>1) {
        ptr_h = Dist_h;
        ptr_d = Dist_d[tid];
        for(int i=0; i<Rounds; i++) {
            if(i%2==tid) {
                bline = i==Rounds-1 ? last_line : block_size;
                cudaMemcpy2DAsync(ptr_h, n_bytes, ptr_d, pitch_bytes, n_bytes, bline, cudaMemcpyDeviceToHost, stream[tid][i]);
                //fprintf(stderr, "memcpy %d %s\n", i, cudaGetErrorString(e));
            }
            ptr_h += line_n;
            ptr_d += line_d;
        }
    } else if(tid==0) {
        cudaMemcpy2DAsync(Dist_h, n_bytes, Dist_d[tid], pitch_bytes, n_bytes, n, cudaMemcpyDeviceToHost);
    }
#pragma omp barrier

    gettimeofday(&temp_time, NULL);
    fprintf(stderr, "before output> %g s\n", CAL_TIME);
#pragma omp sections
    {
#pragma omp section
        {
            for(int d=0; d<max_devices; d++) {
                cudaSetDevice(d);
                cudaDeviceSynchronize();
            }
            //fprintf(stderr, "%d - write file\n", tid);
            msync(Dist_h, out_size, MS_SYNC);
            munmap(Dist_h, out_size);
            fclose(outfile);

            gettimeofday(&temp_time, NULL);
            fprintf(stderr, "\twrite done> %g s\n", CAL_TIME);
        }
#pragma omp section
        {
            //fprintf(stderr, "%d - cleanup\n", tid);
            cudaSetDevice(0);
            cudaDeviceSynchronize();
            cudaStreamDestroy(stream_m[0]);
            for(int i=0; i<Rounds; i++) {
                cudaStreamDestroy(stream[0][i]);
            }
            cudaEventDestroy(ev_1[0]);
            cudaEventDestroy(ev_2[0]);
            cudaEventDestroy(ev_m[0]);
            cudaFree(Dist_d[0]);

            gettimeofday(&temp_time, NULL);
            fprintf(stderr, "\tcuda free done (0)> %g s\n", CAL_TIME);
        }
#pragma omp section
        {
            //fprintf(stderr, "%d - cleanup\n", tid);
            cudaSetDevice(1);
            cudaDeviceSynchronize();
            cudaStreamDestroy(stream_m[1]);
            for(int i=0; i<Rounds; i++) {
                cudaStreamDestroy(stream[1][i]);
            }
            cudaEventDestroy(ev_1[1]);
            cudaEventDestroy(ev_2[1]);
            cudaEventDestroy(ev_m[1]);
            cudaFree(Dist_d[1]);

            gettimeofday(&temp_time, NULL);
            fprintf(stderr, "\tcuda free done (1)> %g s\n", CAL_TIME);
        }
    }
}
