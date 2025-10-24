#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char *argv[]) {
    int rank, size;
    long n;
    double my_pi, pi;

    // no MPI calls before this point
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get synchronized start time
    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();

    if (rank == 0) {
        n = 10000000000;
    }

    MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    const double h = 1.0 / (double)n;
    double sum = 0.0;

    for (long i = rank; i < n; i += size) {
        const double x = h * ((double)i + 0.5);
        sum += 4.0 / (1.0 + x * x);
    }
    my_pi = h * sum;

    MPI_Reduce(&my_pi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Get end time
    const double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Running on %d processes.\n", size);
        printf("Calculated pi = %.16f\n", pi);
        printf("Reference pi  = %.16f\n", M_PI);
        printf("Error         = %.16f\n", fabs(pi - M_PI));
        printf("Time elapsed  = %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    // no MPI calls after this point
    return 0;
}
