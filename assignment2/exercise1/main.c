#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <tgmath.h>

#define HELP_ARG_KEY  "-h"
#define NUM_THREADS_ARG_KEY  "--num-threads"
#define NUM_TRAPEZES_ARG_KEY "--num-trapezes"
#define USAGE_STR "Usage:\n" \
	"\t´" HELP_ARG_KEY "´ show this message\n" \
	"\t´" NUM_THREADS_ARG_KEY " <n>´ use `n` (non zero) threads for calculation. (default: system cpu count)\n" \
	"\t´" NUM_TRAPEZES_ARG_KEY " <n>´ use `n` (non zero) trapezes for calculation. (required)\n"

#define NUMERATOR       (double)4
#define DENOM_LEFT_TERM (double)1
#define INTERVAL_START  (double)0
#define INTERVAL_END    (double)1

typedef int error_t;
#define ERR_SUCCESS       0
#define ERR_NULL_PTR      1
#define ERR_MALLOC        2
#define ERR_DUPLICATE_ARG 3
#define ERR_SYNTAX        4
#define ERR_OUT_OF_RANGE  5
#define ERR_SYSTEM        6

typedef struct config {
	bool help_requested;
	uint32_t num_threads;
	uint64_t num_trapezes;
} config_t;

typedef struct worker_context {
	pthread_t thread;
	uint64_t num_trapezes;
	double start;
	double step;
	double result;
} worker_context_t;

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
	case ERR_OUT_OF_RANGE:
		fprintf(stderr, "value out of range\n");
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
			char *end_ptr;
			errno = 0;
			const unsigned long long num_threads_input = strtoull(argv[i+1], &end_ptr, 10);
			if (errno == ERANGE || num_threads_input > UINT32_MAX) {
				return ERR_OUT_OF_RANGE;
			}
			if (*end_ptr != '\0') {
				return ERR_SYNTAX;
			}
			if (num_threads_input == 0) {
				return ERR_SYNTAX;
			}
			cfg->num_threads = (uint32_t)num_threads_input;
		}

		if (strcmp(argv[i], NUM_TRAPEZES_ARG_KEY) == 0) {
			if (i + 1 >= argc) {
				return ERR_SYNTAX;
			}
			if (cfg->num_trapezes != 0) {
				return ERR_DUPLICATE_ARG;
			}
			char *end_ptr;
			errno = 0;
			const unsigned long long num_trapezes_input = strtoull(argv[i+1], &end_ptr, 10);
			if (errno == ERANGE) {
				return ERR_OUT_OF_RANGE;
			}
			if (*end_ptr != '\0') {
				return ERR_SYNTAX;
			}
			if (num_trapezes_input == 0) {
				return ERR_SYNTAX;
			}
			cfg->num_trapezes = (uint64_t)num_trapezes_input;
		}
	}

	if (cfg->num_threads == 0) {
		long n_procs = sysconf(_SC_NPROCESSORS_ONLN);
		if (n_procs == -1) {
			return ERR_SYSTEM;
		}
		cfg->num_threads = n_procs;
	}

	if (cfg->num_trapezes == 0) {
		return ERR_SYNTAX;
	}

	return ERR_SUCCESS;
}

static double f(const double x) {
	return NUMERATOR / (DENOM_LEFT_TERM + x * x);
}

static void *worker(void *in_data) {
	worker_context_t *ctx = in_data;

	double left_edge = f(ctx->start);
	for (uint64_t i = 1; i <= ctx->num_trapezes; ++i) {
		const double right_edge = f(ctx->start + (double)i * ctx->step);

		// Add the rectangle:
		// After visual analysis in a graphing calculator,
		// the right edge should always be smaller.
		ctx->result += right_edge * ctx->step;

		// Add the triangle:
		// Same assumption as above.
		ctx->result += (left_edge - right_edge) * ctx->step / 2;

		left_edge = right_edge;
	}

	pthread_exit(NULL);
}

static void distribute_trapezes(worker_context_t *workers, const uint32_t num_workers, const uint64_t total_trap) {
	const uint64_t base = total_trap / num_workers;
	const uint64_t remainder = total_trap % num_workers;

	for (uint32_t i = 0; i < num_workers; i++) {
		if (i < remainder) {
			workers[i].num_trapezes = base + 1;
		} else {
			workers[i].num_trapezes = base;
		}
	}
}

static void assign_start_step(worker_context_t *workers, const uint32_t num_workers, const uint64_t total_trap) {
	const double step = (INTERVAL_END - INTERVAL_START) / (double)total_trap;
	double current_start = INTERVAL_START;

	for (uint32_t i = 0; i < num_workers; i++) {
		workers[i].step = step;
		workers[i].start = current_start;
		current_start += (double)workers[i].num_trapezes * step;
	}
}

static double get_total_result(const worker_context_t *workers, const uint32_t num_workers) {
	double total = 0;
	for (uint32_t i = 0; i < num_workers; i++) {
		total += workers[i].result;
	}
	return total;
}

int main(const int argc, char **argv) {
	config_t cfg = {0};

	const error_t err = get_config(argc, argv, &cfg);
	if (err != ERR_SUCCESS) {
		fprintf(stderr, "fatal: ");
		print_error(err);
		fprintf(stderr, USAGE_STR);
		exit(EXIT_FAILURE);
	}
	if (cfg.help_requested) {
		must_printf(USAGE_STR);
		exit(EXIT_SUCCESS);
	}

	worker_context_t *workers = calloc(cfg.num_threads, sizeof(worker_context_t));
	if (workers == NULL) {
		fprintf(stderr, "fatal: ");
		print_error(ERR_MALLOC);
		exit(EXIT_FAILURE);
	}
	distribute_trapezes(workers, cfg.num_threads, cfg.num_trapezes);
	assign_start_step(workers, cfg.num_threads, cfg.num_trapezes);

	// Start time measurement
	struct timespec time_start, time_end;
	clock_gettime(CLOCK_MONOTONIC, &time_start);

	for (uint32_t i = 0; i < cfg.num_threads; ++i) {
		const int ret = pthread_create(&workers[i].thread, NULL, worker, &workers[i]);
		if (ret != 0) {
			fprintf(stderr, "fatal: ");
			print_error(ERR_SYSTEM);
		}
	}

	for (uint32_t i = 0; i < cfg.num_threads; ++i) {
		const int ret = pthread_join(workers[i].thread, NULL);
		if (ret != 0) {
			fprintf(stderr, "fatal: ");
			print_error(ERR_SYSTEM);
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

	const double total = get_total_result(workers, cfg.num_threads);
	must_printf("Results: \n"
		"\tNumber of trapezes: %" PRId64 "\n"
		"\tNumber of threads: %" PRId32 "\n"
		"\tCalculated value: %.17f\n"
		"\t17 digits of pi:  3.14159265358979323\n"
		"\tCalculation time (real): %ld.%09ld seconds\n",
		cfg.num_trapezes, cfg.num_threads, total, time_sec, time_nsec);

	exit(EXIT_SUCCESS);
}
