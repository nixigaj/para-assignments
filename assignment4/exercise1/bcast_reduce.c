#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define HELP_ARG_KEY        "-h"
#define MAX_NUMBER_ARG_KEY  "--max-number"
#define USAGE_STR "Usage:\n" \
    "\t`" HELP_ARG_KEY "` show this message\n" \
    "\t`" MAX_NUMBER_ARG_KEY " <n>` find all primes up to `n` (non zero). (required)\n"

typedef int error_t;
#define ERR_SUCCESS       0
#define ERR_NULL_PTR      1
#define ERR_MALLOC        2
#define ERR_DUPLICATE_ARG 3
#define ERR_SYNTAX        4
#define ERR_SYSTEM        5

#define ROOT_RANK 0

typedef struct config {
    bool help_requested;
    uint64_t max_number;
} config_t;

static void must_printf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    const int ret = vprintf(fmt, args);
    va_end(args);
    if (ret < 0) {
        fprintf(stderr, "fatal: write to standard output failed\n");
        exit(EXIT_FAILURE);
    }
}

static void print_error(const error_t err) {
    switch (err) {
    case ERR_MALLOC:
        fprintf(stderr, "failed to allocate heap memory\n");
        return;
    case ERR_DUPLICATE_ARG:
        fprintf(stderr, "duplicate command line arguments\n");
        return;
    case ERR_SYNTAX:
        fprintf(stderr, "syntax error\n");
        return;
    case ERR_SYSTEM:
        fprintf(stderr, "system error\n");
        return;
    default:
        fprintf(stderr, "unknown error\n");
    }
}

static error_t get_config(const int argc, char **argv, config_t *cfg) {
    for (uint32_t i = 1; i < argc; ++i) {
        if (strcmp(argv[i], HELP_ARG_KEY) == 0) {
            cfg->help_requested = true;
            return ERR_SUCCESS;
        }
        if (strcmp(argv[i], MAX_NUMBER_ARG_KEY) == 0) {
            if (i + 1 >= argc) {
                return ERR_SYNTAX;
            }
            if (cfg->max_number != 0) {
                return ERR_DUPLICATE_ARG;
            }
            if (argv[i+1][0] == '-') {
                return ERR_SYNTAX;
            }
            char *end_ptr;
            errno = 0;
            const unsigned long long max_num_input = strtoull(argv[i+1], &end_ptr, 10);
            if (errno == ERANGE || *end_ptr != '\0' || max_num_input == 0) {
                return ERR_SYNTAX;
            }
            cfg->max_number = (uint64_t)max_num_input;
            i++;
        }
    }
    if (cfg->max_number == 0 && !cfg->help_requested) {
        return ERR_SYNTAX;
    }
    return ERR_SUCCESS;
}

static uint64_t count_primes_in_chunk(const bool *sieve, const uint64_t chunk_size) {
    uint64_t count = 0;
    for (uint64_t i = 0; i < chunk_size; ++i) {
        if (sieve[i]) {
            count++;
        }
    }
    return count;
}

static uint64_t count_primes_in_seeds(const bool *seed_sieve, const uint64_t max_val) {
     uint64_t count = 0;
    for (uint64_t i = 2; i <= max_val; ++i) {
        if (seed_sieve[i]) {
            count++;
        }
    }
    return count;
}


int main(int argc, char **argv) {
    // MPI init
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    config_t cfg = {0};
    struct timespec time_start, time_end;

    // Conf. and broadcast
    if (mpi_rank == ROOT_RANK) {
        const error_t err = get_config(argc, argv, &cfg);
        if (err != ERR_SUCCESS) {
            fprintf(stderr, "fatal: ");
            print_error(err);
            fprintf(stderr, "\n%s", USAGE_STR);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            exit(EXIT_FAILURE);
        }
        if (cfg.help_requested) {
            must_printf(USAGE_STR);
            MPI_Abort(MPI_COMM_WORLD, EXIT_SUCCESS);
            exit(EXIT_FAILURE);
        }

        // Start time measurement
        clock_gettime(CLOCK_MONOTONIC, &time_start);
    }

    MPI_Bcast(&cfg.max_number, 1, MPI_UINT64_T, ROOT_RANK, MPI_COMM_WORLD);

    // All processes compute the sequential seed primes
    const uint64_t seq_max = (uint64_t)sqrt((double)cfg.max_number);
    bool *seed_sieve = malloc((seq_max + 1) * sizeof(bool));
    if (seed_sieve == NULL) {
        fprintf(stderr, "rank %d fatal: malloc for seed_sieve failed\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }

    for (uint64_t i = 0; i <= seq_max; ++i) {
        seed_sieve[i] = true;
    }
    seed_sieve[0] = seed_sieve[1] = false;

    // Sequential seed primes
    for (uint64_t k = 2; k * k <= seq_max; ++k) {
        if (seed_sieve[k]) {
            for (uint64_t k2 = k * k; k2 <= seq_max; k2 += k) {
                seed_sieve[k2] = false;
            }
        }
    }

    uint64_t local_count = 0;
    bool *local_chunk_sieve = NULL;

    const uint64_t parallel_start = seq_max + 1;

    // Handle edge case where no parallel work is needed
    if (parallel_start > cfg.max_number) {
        if (mpi_rank == ROOT_RANK) {
            local_count = count_primes_in_seeds(seed_sieve, cfg.max_number);
        }
    } else {
        // Non edge-case path
        const uint64_t range_size = cfg.max_number - parallel_start + 1;
        const uint64_t base_chunk = range_size / mpi_size;
        const uint64_t remainder = range_size % mpi_size;

        uint64_t local_chunk_size;
        if (mpi_rank < remainder) {
            local_chunk_size = base_chunk + 1;
        } else {
            local_chunk_size = base_chunk + 0;
        }
        const uint64_t my_chunk_start = parallel_start + mpi_rank * base_chunk + (mpi_rank < remainder ? mpi_rank : remainder);

        // Allocate for local sieve
        local_chunk_sieve = malloc(local_chunk_size * sizeof(bool));
        if (local_chunk_sieve == NULL) {
            fprintf(stderr, "rank %d fatal: malloc for my_chunk_sieve failed\n", mpi_rank);
            free(seed_sieve);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            exit(EXIT_FAILURE);
        }
        for (uint64_t i = 0; i < local_chunk_size; ++i) local_chunk_sieve[i] = true;

        // Sieve calculation
        for (uint64_t p = 2; p <= seq_max; ++p) {
            if (seed_sieve[p]) {
                // Find the first multiple of p within this chunk
                uint64_t start_num = (my_chunk_start / p) * p;
                if (start_num < my_chunk_start) {
                    start_num += p;
                }

                // Convert the number to an index in the local chunk array
                // Handle case where start_num might be == 0
                uint64_t start_index;
                if (start_num >= my_chunk_start) {
                    start_index = start_num - my_chunk_start;
                } else {
                    start_index = 0;
                }

                // Mark all multiples of p in local chunk
                for (uint64_t i = start_index; i < local_chunk_size; i += p) {
                    local_chunk_sieve[i] = false;
                }
            }
        }

        local_count = count_primes_in_chunk(local_chunk_sieve, local_chunk_size);

        if (mpi_rank == ROOT_RANK) {
            local_count += count_primes_in_seeds(seed_sieve, seq_max);
        }
    }

    uint64_t total_primes = 0;
    MPI_Reduce(
        &local_count,
        &total_primes,
        1,
        MPI_UINT64_T,
        MPI_SUM,
        ROOT_RANK,
        MPI_COMM_WORLD
    );


    if (mpi_rank == ROOT_RANK) {
        clock_gettime(CLOCK_MONOTONIC, &time_end);
        long time_sec = time_end.tv_sec - time_start.tv_sec;
        long time_nsec = time_end.tv_nsec - time_start.tv_nsec;
        if (time_nsec < 0) {
            time_sec -= 1;
            time_nsec += 1000000000L;
        }

        must_printf("Results:\n"
            "\tPrimes up to: %" PRIu64 "\n"
            "\tNumber of processes: %d\n"
            "\tPrimes found: %" PRIu64 "\n"
            "\tCalculation time (real): %ld.%09ld seconds\n",
            cfg.max_number, mpi_size, total_primes, time_sec, time_nsec);
    }

    free(seed_sieve);
    free(local_chunk_sieve);
    MPI_Finalize();
    exit(EXIT_SUCCESS);
}
