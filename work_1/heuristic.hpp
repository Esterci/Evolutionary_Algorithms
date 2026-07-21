

struct solution
{
  int *tour; // this is what the fitness_evaluation function in EVRP.hpp will evaluate
  int id;
  double tour_length; // quality of the solution
  int steps;          // size of the solution
  double weight;
  int *cromossome;
  // the format of the solution is as follows:
  //*tour:  0 - 5 - 6 - 8 - 0 - 1 - 2 - 3 - 4 - 0 - 7 - 0
  //*steps: 12
  // this solution consists of three routes:
  // Route 1: 0 - 5 - 6 - 8 - 0
  // Route 2: 0 - 1 - 2 - 3 - 4 - 0
  // Route 3: 0 - 7 - 0
};

extern solution *best_sol;
extern int n_pop;

bool compare_fitness(const solution &a, const solution &b);

double linear_classification(int i);

void take_route(solution *route);

void initialize_heuristic(int run);

int parent_selection(solution ranked[]);

void crossover(int p1, int p2);

void change_pop();

void run_heuristic();

void free_heuristic();