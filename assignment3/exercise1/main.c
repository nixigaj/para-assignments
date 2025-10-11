#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define HELP_ARG_KEY        "-h"
#define NUM_THREADS_ARG_KEY "--num-threads"
#define MAX_NUMBER_ARG_KEY  "--max-number"
#define USAGE_STR "Usage:\n" \
    "\t`" HELP_ARG_KEY "` show this message\n" \
    "\t`" NUM_THREADS_ARG_KEY " <n>` use `n` (non zero) threads for calculation. (default: system cpu count)\n" \
    "\t`" MAX_NUMBER_ARG_KEY " <n>` find all primes up to `n` (non zero). (required)\n"

typedef int error_t;
#define ERR_SUCCESS       0
#define ERR_NULL_PTR      1
#define ERR_MALLOC        2
#define ERR_DUPLICATE_ARG 3
#define ERR_SYNTAX        4
#define ERR_SYSTEM        5

typedef struct config {
    bool help_requested;
    uint32_t num_threads;
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
        if (strcmp(argv[i], NUM_THREADS_ARG_KEY) == 0) {
            if (i + 1 >= argc) {
                return ERR_SYNTAX;
            }
            if (cfg->num_threads != 0) {
                return ERR_DUPLICATE_ARG;
            }
            if (argv[i+1][0] == '-') {
                return ERR_SYNTAX;
            }
            char *end_ptr;
            errno = 0;
            const unsigned long long num_threads_input = strtoull(argv[i+1], &end_ptr, 10);
            if (errno == ERANGE || num_threads_input > UINT32_MAX || *end_ptr != '\0' || num_threads_input == 0) {
                return ERR_SYNTAX;
            }
            cfg->num_threads = (uint32_t)num_threads_input;
            i++;
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
    if (cfg->num_threads == 0) {
        const long n_procs = sysconf(_SC_NPROCESSORS_ONLN);
        if (n_procs == -1) return ERR_SYSTEM;
        cfg->num_threads = n_procs;
    }
    if (cfg->max_number == 0) return ERR_SYNTAX;
    return ERR_SUCCESS;
}

static uint64_t count_primes(const bool *is_prime, const uint64_t max_number) {
    uint64_t count = 0;
    for (uint64_t i = 2; i <= max_number; ++i) {
        if (is_prime[i]) {
            count++;
        }
    }
    return count;
}

int main(const int argc, char **argv) {
    config_t cfg = {0};

    const error_t err = get_config(argc, argv, &cfg);
    if (err != ERR_SUCCESS) {
        fprintf(stderr, "fatal: ");
        print_error(err);
        fprintf(stderr, "\n%s", USAGE_STR);
        exit(EXIT_FAILURE);
    }
    if (cfg.help_requested) {
        must_printf(USAGE_STR);
        exit(EXIT_SUCCESS);
    }

    bool *is_prime = malloc((cfg.max_number + 1) * sizeof(bool));
    if (is_prime == NULL) {
        fprintf(stderr, "fatal: ");
        print_error(ERR_MALLOC);
        exit(EXIT_FAILURE);
    }
    // All entries are assumed primes by default
    for (uint64_t i = 0; i <= cfg.max_number; ++i) {
        is_prime[i] = true;
    }
    is_prime[0] = is_prime[1] = false;

    omp_set_num_threads((int)cfg.num_threads);

    // Start time measurement
    struct timespec time_start, time_end;
    clock_gettime(CLOCK_MONOTONIC, &time_start);

    // Sequential seed primes
    const uint64_t seq_max = (uint64_t) sqrt((double) cfg.max_number);
    for (uint64_t k = 2; k * k <= seq_max; ++k) {
        if (is_prime[k]) {
            for (uint64_t k2 = k * k; k2 <= seq_max; k2 += k) {
                is_prime[k2] = false;
            }
        }
    }

    // New parallel calculation, moved out of worker function
    #pragma omp parallel for schedule(dynamic) default(none) shared(is_prime, seq_max, cfg)
    for (uint64_t k = 2; k <= seq_max; k++) {
        if (is_prime[k]) {
            for (uint64_t k2 = k * k; k2 <= cfg.max_number; k2 += k) {
                is_prime[k2] = false;
            }
        }
    }

    // End time measurement
    clock_gettime(CLOCK_MONOTONIC, &time_end);
    long time_sec = time_end.tv_sec - time_start.tv_sec;
    long time_nsec = time_end.tv_nsec - time_start.tv_nsec;
    if (time_nsec < 0) {
        time_sec -= 1;
        time_nsec += 1000000000L;
    }

    // The is_prime array can also be dumped to a
    // binary file if one actually wants to save the result.
    // This is just a sanity check
    const uint64_t total_primes = count_primes(is_prime, cfg.max_number);

    must_printf("Results:\n"
        "\tPrimes up to: %" PRIu64 "\n"
        "\tNumber of threads: %" PRIu32 "\n"
        "\tPrimes found: %" PRIu64 "\n"
        "\tCalculation time (real): %ld.%09ld seconds\n",
        cfg.max_number, cfg.num_threads, total_primes, time_sec, time_nsec);

    free(is_prime);
    exit(EXIT_SUCCESS);
}
