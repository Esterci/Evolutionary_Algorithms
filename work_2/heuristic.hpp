#include <string>

struct solution
{
  int *tour; // this is what the fitness_evaluation function in EVRP.hpp will evaluate
  int id;
  double tour_length; // quality of the solution
  int steps;          // size of the solution
  double weight;
  int *cromossome;
  bool viable;
  double penalty;
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
extern float mutation_rate;
extern float crossover_rate;
extern std::string mutation_method;
extern float select_pres;
extern double fitness_mean;
extern double fitness_std;

bool compare_fitness(const solution &a, const solution &b);

/*
 * Checks whether the node corresponds to a customer.
 *
 * Assumes:
 *   customers: 1 to NUM_OF_CUSTOMERS
 *   depot: DEPOT
 *   charging stations: above NUM_OF_CUSTOMERS
 */
bool is_customer_node(int node);

/*
 * Checks whether the vehicle can travel from "from" to "to"
 * considering the energy already consumed since the last recharge.
 */
bool can_reach_node(int from, int to, double energy_used);

/*
 * Returns the minimum amount of energy required to leave a node
 * and reach a location where the battery can be recharged.
 *
 * The recharge location may be:
 *   - the depot;
 *   - a charging station.
 */
double minimum_energy(int from);

/*
 * Finds the nearest charging station that:
 *
 * 1. can be reached using the current battery charge;
 * 2. allows the vehicle to reach the customer after recharging;
 * 3. allows the vehicle to leave the customer and reach another
 *    charging station or the depot.
 *
 * Therefore, this function does not simply select the nearest station.
 * It selects the nearest station that keeps the route feasible.
 */
int nearest_stat_bef_customer(int from, int customer, double energy_used);

/*
 * Finds the nearest charging station that:
 *
 * 1. can be reached using the current battery charge;
 * 2. allows the vehicle to reach the depot after recharging.
 *
 * This function is used when the vehicle capacity is exceeded
 * or when the final route must be closed.
 */
int nearest_stat_bef_depot(int from, double energy_used);

/*
 * Recomputes the used capacity and consumed energy
 * after a backtracking operation.
 *
 * This avoids storing an additional array containing
 * the state associated with each route step.
 */
void recompute_route_state(const solution *route, double &capacity_used, double &energy_used);

/*
 * Removes the last customer inserted into the route.
 *
 * The chromosome index is also decremented so that the removed
 * customer can be processed again, this time with a charging
 * station inserted before it.
 */
bool rollback(solution *route, int &customer_index, double &capacity_used, double &energy_used);

/*
 * Builds a "feasible" (I hope) route.
 */
void take_route(solution *route);

int hamming_distance(const int *vector1, const int *vector2, int dist_limit);

double linear_classification(int i);

void get_fitness_mean();

void get_fitness_std();

void initialize_heuristic(int run);

int parent_selection(solution ranked[]);

void crossover(int p1, int p2);

void copy_solution(solution &destination, const solution &source);

void change_pop();

// Mutação por inserção
void mutation_insertion();

// Mutação por troca
void mutation_swap();

// Mutação por mistura
void mutation_mix();

// Mutação por inversão
void mutation_inversion();

void printChromosome();

void run_heuristic();

void free_heuristic();