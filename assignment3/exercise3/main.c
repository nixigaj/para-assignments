#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(const int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,"Usage: %s <dim> <case (1|2|3|4)> <bin-out-filename (optional)>\n", argv[0]);
    	exit(1);
    }

    int dim = atoi(argv[1]);
    const int variant = atoi(argv[2]);

	double *A = malloc(sizeof(double) * (size_t)dim * (size_t)dim);
	if (A == NULL) {
		fprintf(stderr,"error: failed to allocate memory\n");
		free(A);
		exit(1);
	}
	double *B = malloc(sizeof(double) * (size_t)dim * (size_t)dim);
	if (B == NULL) {
		fprintf(stderr,"error: failed to allocate memory\n");
		free(A);
		free(B);
		exit(1);
	}
	double *C = malloc(sizeof(double) * (size_t)dim * (size_t)dim);
	if( C == NULL) {
		fprintf(stderr,"error: failed to allocate memory\n");
		free(A);
		free(B);
		free(C);
		exit(1);
	}

	// This is not used for cryptography so we can use a deterministic seed
    unsigned seed = 12345;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            A[i * (size_t)dim + j] = (double)rand_r(&seed) / RAND_MAX;
            B[i * (size_t)dim + j] = (double)rand_r(&seed) / RAND_MAX;
            C[i * (size_t)dim + j] = 0.0;
        }
    }

	// Zero out C
	for (int i = 0; i < dim * (size_t)dim; i++) {
		C[i] = 0.0;
	}

	struct timespec ts_start;
	clock_gettime(CLOCK_MONOTONIC, &ts_start);

	if (variant == 1) {
        // outermost loop
        int i, j, k;

        #pragma omp parallel for schedule(static) collapse(1) default(none) shared(A, B, C, dim) private(i, j, k)
        for (i = 0; i < dim; i++) {
            for (j = 0; j < dim; j++) {
                double sum = 0.0;
                for (k = 0; k < dim; k++) {
                    sum += A[i * (size_t)dim + k] * B[k * (size_t)dim + j];
                }
                C[i * (size_t)dim + j] = sum;
            }
        }
    } else if (variant == 2) {
        // outer two loops collapsed
        int i, j, k;

        #pragma omp parallel for schedule(static) collapse(2) default(none) shared(A, B, C, dim) private(i, j, k)
        for (i = 0; i < dim; i++) {
            for (j = 0; j < dim; j++) {
                double sum = 0.0;
                for(k=0; k<dim; k++) {
                    sum += A[i * (size_t)dim + k] * B[k * (size_t)dim + j];
                }
                C[i * (size_t)dim + j] = sum;
            }
        }
    } else if (variant == 3) {
        // all three loops collapsed with atomic safety
        int i, j, k;

        #pragma omp parallel for schedule(static) collapse(3) default(none) shared(A, B, C, dim) private(i, j, k)
        for (i=0; i<dim; i++){
            for (j = 0; j < dim; j++) {
                for (k = 0; k < dim; k++) {
                    const size_t idx = (size_t)i * (size_t)dim + (size_t)j;
                    const double prod = A[i * (size_t)dim + k] * B[k * (size_t)dim + j];
                    #pragma omp atomic
                    C[idx] += prod;
                }
            }
        }
    } else if (variant == 4) {
    	// all three loops collapsed without atomic safety
    	int i, j, k;

		#pragma omp parallel for schedule(static) collapse(3) default(none) shared(A, B, C, dim) private(i, j, k)
    	for (i = 0; i < dim; i++) {
    		for (j = 0; j < dim; j++) {
    			double sum = 0.0;
    			for(k=0; k<dim; k++) {
    				sum += A[i * (size_t)dim + k] * B[k * (size_t)dim + j];
    			}
    			C[i * (size_t)dim + j] = sum;
    		}
    	}
    } else {
        fprintf(stderr,"error: unknown variant\n");
        free(A);
        free(B);
        free(C);
        exit(1);
    }

	struct timespec ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    long sec = ts_end.tv_sec - ts_start.tv_sec;
    long nsec = ts_end.tv_nsec - ts_start.tv_nsec;
    if (nsec < 0) {
        sec -= 1; nsec += 1000000000L;
    }
    const double elapsed = (double)sec + (double)nsec * 1e-9;

    printf("time elapsed: %.6f seconds\n", elapsed);

	if (argc >= 4) {
		char *filename = argv[3];
		FILE *fp = fopen(filename, "wb");
		if (fp == NULL) {
			fprintf(stderr, "error: could not open %s for writing\n", filename);
		} else {
			const size_t written = fwrite(C, sizeof(double), (size_t)dim * (size_t)dim, fp);
			if (written != (size_t)dim * (size_t)dim) {
				fprintf(stderr, "warning: incomplete write to %s (%zu/%zu elements)\n",
						filename, written, (size_t)dim * (size_t)dim);
			}
			fclose(fp);
			printf("wrote %zu doubles to %s\n", written, filename);
		}
	}

    free(A);
    free(B);
    free(C);
    exit(0);
}
