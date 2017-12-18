#include <stdio.h>
#include <stdlib.h>

#define INF 1000000000
#define V   20500


void input(char *inFileName, int Dist[V][V], int *n)
{
    int m;
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", n, &m);

	for (int i = 0; i < *n; ++i) {
		for (int j = 0; j < *n; ++j) {
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

void output(char *outFileName, int Dist[V][V], int n)
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
