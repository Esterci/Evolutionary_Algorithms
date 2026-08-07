#define MAX_TRIALS 20 // DO NOT CHANGE THE NUMBER
#define CHAR_LEN 100

extern FILE *log_performance;
extern FILE *log_curves;

void open_stats(void);              // creates the output file
void close_stats(void);             // stores the best values for each run
void get_mean(int r, double value); // stores the observation from each run
void free_stats();                  // free memory
void open_curves();
void write_solution(int *routes, int size, bool viable);
void write_curves(int seed, int it, double fit_mean, double fit_std);
void close_curves(void);