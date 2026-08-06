#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <math.h>
#include <fstream>
#include <limits.h>
#include <algorithm>
#include <vector>

#include "heuristic.hpp"
#include "EVRP.hpp"

using namespace std;

solution *best_sol; // see heuristic.hpp for the solution structure
solution *population;
solution *offspring;

// Global variables
int n_pop;
float mutation_rate;
float crossover_rate;
string mutation_method;
float select_pres;
double fitness_mean;
double fitness_std;

// Context variables
int **lv_distance;

const double EPS = 1e-9;

bool compare_fitness(const solution &a, const solution &b)
{
  return a.tour_length > b.tour_length;
}

/*
 * Checks whether the node corresponds to a customer.
 *
 * Assumes:
 *   customers: 1 to NUM_OF_CUSTOMERS
 *   depot: DEPOT
 *   charging stations: above NUM_OF_CUSTOMERS
 */
bool is_customer_node(int node)
{
  return node >= 1 && node <= NUM_OF_CUSTOMERS;
}

/*
 * Checks whether the vehicle can travel from "from" to "to"
 * considering the energy already consumed since the last recharge.
 */
bool can_reach_node(int from, int to, double energy_used)
{
  double consumption = get_energy_consumption(from, to);

  return energy_used + consumption <= BATTERY_CAPACITY + EPS;
}

/*
 * Returns the minimum amount of energy required to leave a node
 * and reach a location where the battery can be recharged.
 *
 * The recharge location may be:
 *   - the depot;
 *   - a charging station.
 */
double minimum_energy(int from)
{
  double minimum_energy =
      get_energy_consumption(from, DEPOT);

  for (int station = NUM_OF_CUSTOMERS + 1;
       station < ACTUAL_PROBLEM_SIZE;
       station++)
  {
    if (!is_charging_station(station))
      continue;

    if (station == from)
      continue;

    double consumption =
        get_energy_consumption(from, station);

    if (consumption < minimum_energy)
    {
      minimum_energy = consumption;
    }
  }

  return minimum_energy;
}

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
int nearest_stat_bef_customer(int from, int customer, double energy_used)
{
  int best_station = -1;

  double best_distance =
      std::numeric_limits<double>::infinity();

  double escape_energy =
      minimum_energy(customer);

  for (int station = NUM_OF_CUSTOMERS + 1;
       station < ACTUAL_PROBLEM_SIZE;
       station++)
  {
    if (!is_charging_station(station))
      continue;

    if (station == from)
      continue;

    double energy_to_station =
        get_energy_consumption(from, station);

    /*
     * First, check whether the station can be reached
     * using the current battery charge.
     */
    if (energy_used + energy_to_station >
        BATTERY_CAPACITY + EPS)
    {
      continue;
    }

    double station_to_customer =
        get_energy_consumption(station, customer);

    /*
     * After recharging, it must be possible to travel:
     *
     * station -> customer -> depot/station
     */
    if (station_to_customer + escape_energy >
        BATTERY_CAPACITY + EPS)
    {
      continue;
    }

    /*
     * Among the feasible charging stations, select the one
     * closest to the current position.
     */
    if (energy_to_station < best_distance)
    {
      best_distance = energy_to_station;
      best_station = station;
    }
  }

  return best_station;
}

/*
 * Finds the nearest charging station that:
 *
 * 1. can be reached using the current battery charge;
 * 2. allows the vehicle to reach the depot after recharging.
 *
 * This function is used when the vehicle capacity is exceeded
 * or when the final route must be closed.
 */
int nearest_stat_bef_depot(int from, double energy_used)
{
  int best_station = -1;

  double best_distance =
      std::numeric_limits<double>::infinity();

  for (int station = NUM_OF_CUSTOMERS + 1;
       station < ACTUAL_PROBLEM_SIZE;
       station++)
  {
    if (!is_charging_station(station))
      continue;

    if (station == from)
      continue;

    double energy_to_station =
        get_energy_consumption(from, station);

    /*
     * The station must be reachable using the remaining
     * battery charge.
     */
    if (energy_used + energy_to_station >
        BATTERY_CAPACITY + EPS)
    {
      continue;
    }

    /*
     * After recharging, the vehicle must be able
     * to reach the depot.
     */
    double station_to_depot =
        get_energy_consumption(station, DEPOT);

    if (station_to_depot >
        BATTERY_CAPACITY + EPS)
    {
      continue;
    }

    if (energy_to_station < best_distance)
    {
      best_distance = energy_to_station;
      best_station = station;
    }
  }

  return best_station;
}

/*
 * Recomputes the used capacity and consumed energy
 * after a backtracking operation.
 *
 * This avoids storing an additional array containing
 * the state associated with each route step.
 */
void recompute_route_state(const solution *route, double &capacity_used, double &energy_used)
{
  capacity_used = 0.0;
  energy_used = 0.0;

  for (int step = 1; step < route->steps; step++)
  {
    int previous = route->tour[step - 1];
    int current = route->tour[step];

    energy_used +=
        get_energy_consumption(previous, current);

    /*
     * When the vehicle reaches the depot:
     *   - the vehicle capacity is reset;
     *   - the battery is fully recharged.
     */
    if (current == DEPOT)
    {
      capacity_used = 0.0;
      energy_used = 0.0;
    }
    /*
     * When the vehicle reaches a charging station:
     *   - only the battery is recharged;
     *   - the transported load remains unchanged.
     */
    else if (is_charging_station(current))
    {
      energy_used = 0.0;
    }
    /*
     * When the vehicle reaches a customer:
     *   - the customer's demand is added to the used capacity.
     */
    else if (is_customer_node(current))
    {
      capacity_used +=
          get_customer_demand(current);
    }
  }
}

/*
 * Removes the last customer inserted into the route.
 *
 * The chromosome index is also decremented so that the removed
 * customer can be processed again, this time with a charging
 * station inserted before it.
 */
bool rollback(solution *route, int &customer_index, double &capacity_used, double &energy_used)
{
  if (route->steps <= 1)
  {
    return false;
  }

  if (customer_index <= 0)
  {
    return false;
  }

  int last_node =
      route->tour[route->steps - 1];

  /*
   * The backtracking operation must remove a customer.
   * This function does not remove a depot or charging station.
   */
  if (!is_customer_node(last_node))
  {
    return false;
  }

  route->steps--;

  /*
   * The removed customer must be processed again.
   */
  customer_index--;

  recompute_route_state(
      route,
      capacity_used,
      energy_used);

  return true;
}

/*
 * Builds a feasible route.
 */
void take_route(solution *route)
{
  int customer_index = 0;

  double energy_used = 0.0;
  double capacity_used = 0.0;

  route->steps = 1;
  route->tour_length = INT_MAX;

  route->tour[0] = DEPOT;

  /*
   * Prevents an infinite loop if there is a problem with the
   * instance, chromosome sequence, or charging-station connectivity.
   */
  int iterations = 0;

  const int max_iterations =
      20 * (ACTUAL_PROBLEM_SIZE + NUM_OF_CUSTOMERS);

  while (true)
  {
    iterations++;

    if (iterations > max_iterations)
    {
      route->tour_length = INT_MAX;
      break;
    }

    /*
     * All customers have already been inserted.
     * The route must now be closed at the depot.
     */
    if (customer_index >= NUM_OF_CUSTOMERS)
    {
      int from =
          route->tour[route->steps - 1];

      if (from == DEPOT)
      {
        break;
      }

      /*
       * Return directly to the depot.
       */
      if (can_reach_node(from, DEPOT, energy_used))
      {
        route->tour[route->steps] = DEPOT;
        route->steps++;

        break;
      }

      /*
       * The vehicle cannot reach the depot directly.
       * Find the nearest feasible charging station.
       */
      int station =
          nearest_stat_bef_depot(
              from,
              energy_used);

      if (station != -1)
      {
        route->tour[route->steps] = station;
        route->steps++;

        /*
         * The energy used to reach the charging station does not
         * need to remain stored because the battery is recharged.
         */
        energy_used = 0.0;

        continue;
      }

      /*
       * The vehicle cannot reach the depot or a charging station
       * that would allow it to reach the depot.
       *
       * Remove the last customer and try to recharge before
       * visiting it.
       */
      if (rollback(route, customer_index, capacity_used, energy_used))
      {
        continue;
      }

      route->tour_length = INT_MAX;
      break;
    }

    int from = route->tour[route->steps - 1];

    int customer = route->cromossome[customer_index];

    double customer_demand = get_customer_demand(customer);

    /*
     * A customer whose individual demand exceeds the maximum
     * vehicle capacity can never be served.
     */
    if (customer_demand > MAX_CAPACITY + EPS)
    {
      route->tour_length = INT_MAX;
      break;
    }

    /*
     * The current vehicle capacity is not enough to serve
     * the next customer. The vehicle must return to the depot.
     */
    if (capacity_used + customer_demand > MAX_CAPACITY + EPS)
    {
      /*
       * Return directly to the depot.
       */
      if (can_reach_node(from, DEPOT, energy_used))
      {
        route->tour[route->steps] = DEPOT;
        route->steps++;

        capacity_used = 0.0;
        energy_used = 0.0;

        continue;
      }

      /*
       * There is not enough energy to reach the depot directly.
       * Try to visit a charging station first.
       */
      int station = nearest_stat_bef_depot(from, energy_used);

      if (station != -1)
      {
        route->tour[route->steps] = station;
        route->steps++;

        energy_used = 0.0;

        continue;
      }

      /*
       * The vehicle cannot reach either the depot or a suitable
       * charging station. Backtrack to the previous customer.
       */
      if (rollback(route, customer_index, capacity_used, energy_used))
      {
        continue;
      }

      route->tour_length = INT_MAX;
      break;
    }

    double energy_to_customer = get_energy_consumption(from, customer);

    double energy_after_customer = energy_used + energy_to_customer;

    /*
     * Energy required to reach a charging station or the depot
     * after visiting the customer.
     */
    double escape_energy = minimum_energy(customer);

    /*
     * The customer is inserted only if:
     *
     * 1. the vehicle can reach the customer;
     * 2. the vehicle can still reach a recharge point afterward.
     */
    bool can_visit_customer = energy_after_customer <= BATTERY_CAPACITY + EPS;

    bool can_leave_customer = energy_after_customer + escape_energy <= BATTERY_CAPACITY + EPS;

    if (can_visit_customer && can_leave_customer)
    {
      capacity_used += customer_demand;
      energy_used = energy_after_customer;

      route->tour[route->steps] = customer;
      route->steps++;

      customer_index++;

      continue;
    }

    /*
     * It is not safe to visit the customer using the current
     * battery charge. Find the nearest feasible charging station
     * before visiting the customer.
     */
    int station = nearest_stat_bef_customer(from, customer, energy_used);

    if (station != -1)
    {
      route->tour[route->steps] = station;
      route->steps++;

      energy_used = 0.0;

      continue;
    }

    /*
     * If no suitable charging station is available, the depot
     * may also be used as a recharge point.
     */
    double depot_to_customer = get_energy_consumption(DEPOT, customer);

    bool depot_is_reachable = from != DEPOT && can_reach_node(from, DEPOT, energy_used);

    bool customer_is_safe_from_depot = depot_to_customer + escape_energy <= BATTERY_CAPACITY + EPS;

    if (depot_is_reachable && customer_is_safe_from_depot)
    {
      route->tour[route->steps] = DEPOT;
      route->steps++;

      capacity_used = 0.0;
      energy_used = 0.0;

      continue;
    }

    /*
     * The current battery charge is not enough to safely reach
     * the next customer, and there is no reachable charging
     * station or depot.
     *
     * Remove the previous customer. During the next iteration,
     * that customer will be processed again with a charging
     * station inserted before it.
     */
    if (rollback(route,customer_index,capacity_used,energy_used))
    {
      continue;
    }

    /*
     * There is no previous customer to remove.
     * Therefore, this chromosome cannot be decoded using
     * the current problem configuration.
     */
    route->tour_length = INT_MAX;
    break;
  }

  
}

int hamming_distance(const int *vector1, const int *vector2, int dist_limit)
{
  int dist = 0;

  for (int i = 0; i < NUM_OF_CUSTOMERS; i++)
  {
    if (vector1[i] != vector2[i])
    {
      dist++;

      /*
       * Se a distância já não pode superar a melhor
       * encontrada anteriormente, encerra a comparação.
       */
      if (dist >= dist_limit)
      {
        return dist;
      }
    }
  }

  return dist;
}

double linear_classification(int i)
{
  return (2 - select_pres) / n_pop + 2 * i * (select_pres - 1) / (n_pop * (n_pop - 1));
}

void get_fitness_mean()
{
  int i, total_fitness = 0;

  for (i = 0; i < n_pop; i++)
  {
    total_fitness += population[i].tour_length;
  }

  fitness_mean = total_fitness / n_pop;
}

void get_fitness_std()
{
  int i, total_fitness = 0;

  for (i = 0; i < n_pop; i++)
  {
    total_fitness += pow(population[i].tour_length - fitness_mean, 2);
  }

  fitness_std = sqrt(total_fitness / n_pop);
}

/*initialize the structure of your heuristic in this function*/
void initialize_heuristic(int run)
{
  /*generate a random solution for the random heuristic*/
  int i, j, help, object, tot_assigned;

  // Allocate the Levenshtein distance matrix only once.
  lv_distance = new int *[NUM_OF_CUSTOMERS + 1];

  for (int i = 0; i <= NUM_OF_CUSTOMERS; i++)
  {
    lv_distance[i] =
        new int[NUM_OF_CUSTOMERS + 1];
  }

  // Aloca um vetor com n_pop soluções
  population = new solution[n_pop];
  offspring = new solution[1];

  for (i = 0; i < n_pop; i++)
  {

    population[i].tour = new int[NUM_OF_CUSTOMERS + 1000];
    population[i].cromossome = new int[(NUM_OF_CUSTOMERS)];
    population[i].id = i + 1;
    population[i].steps = 0;
    population[i].tour_length = INT_MAX;
    population[i].weight = 0;

    if (i == 0)
    {
      best_sol = &population[0];
    }

    srand((run - 1) * n_pop + i + 1); // random seed

    tot_assigned = 0;

    for (j = 0; j < NUM_OF_CUSTOMERS; j++)
    {
      population[i].cromossome[j] = j + 1;
    }

    // randomly change indexes of obiects
    for (j = 0; j < NUM_OF_CUSTOMERS; j++) // Tem como fixarmos e salvarmos a seed?
    {
      object = (int)((rand() / (RAND_MAX + 1.0)) * (double)(NUM_OF_CUSTOMERS - tot_assigned));
      help = population[i].cromossome[j];
      population[i].cromossome[j] = population[i].cromossome[j + object];
      population[i].cromossome[j + object] = help;
      tot_assigned++;
    }

    take_route(&population[i]);

    population[i].tour_length = fitness_evaluation(population[i].tour, population[i].steps);

    if (population[i].tour_length < best_sol->tour_length)
    {
      best_sol = &population[i];
    }
  }

  for (i = 0; i < 1; i++)
  {
    offspring[i].tour = new int[NUM_OF_CUSTOMERS + 1000];
    offspring[i].cromossome = new int[(NUM_OF_CUSTOMERS)];
    offspring[i].id = i + 1;
    offspring[i].steps = 0;
    offspring[i].tour_length = INT_MAX;
    offspring[i].weight = 0;
  }
}

int parent_selection(solution ranked[])
{

  // Ordena do menor fitness para o maior.
  sort(ranked, ranked + n_pop, compare_fitness);

  best_sol = &population[n_pop - 1];

  // O melhor recebe o maior peso.
  for (int i = 0; i < n_pop; i++)
  {
    ranked[i].weight = linear_classification(i);
  }

  // Sorteia entre 1 e totalWeight.

  double randomValue = rand() / (RAND_MAX * 1.0);

  double accumulatedWeight = 0;

  for (int i = 0; i < n_pop; i++)
  {
    accumulatedWeight += ranked[i].weight;

    if (randomValue <= accumulatedWeight)
    {
      return i;
    }
  }

  return n_pop - 1;
}

void crossover(int p1, int p2)
{
  int cut1 = rand() % (NUM_OF_CUSTOMERS - 3) + 1;

  int cut2 = rand() % (NUM_OF_CUSTOMERS - 2 - cut1) + (cut1 + 1);

  for (int i = 0; i < NUM_OF_CUSTOMERS; i++)
  {
    offspring[0].cromossome[i] = -1;
  }

  // Copia o segmento do primeiro pai.
  for (int i = cut1; i <= cut2; i++)
  {
    offspring[0].cromossome[i] = population[p1].cromossome[i];
  }

  // Preenche as posições externas usando o segundo pai.
  for (int i = 0; i < NUM_OF_CUSTOMERS; i++)
  {
    if (i < cut1 || i > cut2)
    {
      int value = population[p2].cromossome[i];
      bool conflict = true;

      while (conflict)
      {
        conflict = false;

        // Verifica se o valor já está no segmento copiado.
        for (int j = cut1; j <= cut2; j++)
        {
          if (offspring[0].cromossome[j] == value)
          {
            // Busca o valor correspondente no outro segmento.
            value = population[p2].cromossome[j];
            conflict = true;
            break;
          }
        }
      }

      offspring[0].cromossome[i] = value;
    }
  }
}

void copy_solution(solution &destination, const solution &source)
{
  // Copy the chromosome.
  for (int i = 0; i < NUM_OF_CUSTOMERS; i++)
  {
    destination.cromossome[i] = source.cromossome[i];
  }

  // Copy the number of positions in the complete route.
  destination.steps = source.steps;

  // Copy the complete route.
  for (int i = 0; i < source.steps; i++)
  {
    destination.tour[i] = source.tour[i];
  }

  // Copy the fitness value.
  destination.tour_length = source.tour_length;
}

void change_pop()
{
  int dist;
  int dist_min = INT_MAX;
  int chg_index = -1;

  /*
   * Evaluate the offspring using its complete route.
   */

  offspring[0].tour_length =
      fitness_evaluation(
          offspring[0].tour,
          offspring[0].steps);

  /*
   * Search for the individual whose chromosome is
   * the most similar to the offspring chromosome.
   */
  for (int i = 0; i < n_pop; i++)
  {
    dist = hamming_distance(
        population[i].cromossome,
        offspring[0].cromossome,
        dist_min);

    if (dist < dist_min)
    {
      dist_min = dist;
      chg_index = i;
    }
  }

  /*
   * Deterministic crowding:
   * the offspring competes against the most similar individual.
   * The replacement occurs only if the offspring is better.
   */
  if (offspring[0].tour_length <
      population[chg_index].tour_length)
  {

    copy_solution(
        population[chg_index],
        offspring[0]);
  }

  /*
   * Update the global best solution.
   *
   * best_sol must point to an independently allocated solution.
   * Do not assign best_sol = &offspring[0], because offspring[0]
   * will be overwritten in the next iteration.
   */
  if (offspring[0].tour_length < best_sol->tour_length)
  {
    copy_solution(
        *best_sol,
        offspring[0]);
  }
}

// Mutação por inserção

void mutation_insertion()
{
  int n = NUM_OF_CUSTOMERS;

  if (n < 2)
  {
    return;
  }

  int *chromosome = offspring[0].cromossome;

  // Seleciona a posição original
  int origin = rand() % n;

  // Seleciona a nova posição
  int destination = rand() % n;

  while (destination == origin)
  {
    destination = rand() % n;
  }

  // Guarda o gene que será deslocado
  int gene = chromosome[origin];

  if (origin < destination)
  {
    // Desloca os genes para a esquerda
    for (int i = origin; i < destination; i++)
    {
      chromosome[i] = chromosome[i + 1];
    }
  }
  else
  {
    // Desloca os genes para a direita
    for (int i = origin; i > destination; i--)
    {
      chromosome[i] = chromosome[i - 1];
    }
  }

  // Insere o gene na nova posição
  chromosome[destination] = gene;
}

// Mutação por troca

void mutation_swap()
{
  int aux, g1, g2;

  g1 = rand() % NUM_OF_CUSTOMERS;

  g2 = g1;

  while (g2 == g1)
  {
    g2 = rand() % NUM_OF_CUSTOMERS;
  }

  aux = offspring[0].cromossome[g1];

  offspring[0].cromossome[g1] = offspring[0].cromossome[g2];

  offspring[0].cromossome[g2] = aux;
}

// Mutação por mistura

void mutation_mix()
{
  const int n = NUM_OF_CUSTOMERS;

  if (n < 2)
  {
    return;
  }

  int *chromosome = offspring[0].cromossome;
  vector<int> positions;

  // Quantidade entre 2 e n
  int quantity = 2 + rand() % (n - 1);

  while (
      static_cast<int>(positions.size()) < quantity)
  {
    int position = rand() % n;

    bool repeated =
        find(
            positions.begin(),
            positions.end(),
            position) != positions.end();

    if (!repeated)
    {
      positions.push_back(position);
    }
  }

  // Fisher-Yates nas posições selecionadas
  for (int i = quantity - 1; i > 0; i--)
  {
    int j = rand() % (i + 1);

    swap(
        chromosome[positions[i]],
        chromosome[positions[j]]);
  }
}

// Mutação por inversão

void mutation_inversion()
{
  const int n = NUM_OF_CUSTOMERS;

  if (n < 2)
  {
    return;
  }

  int *chromosome = offspring[0].cromossome;

  int begin_position = rand() % n;
  int end_position = rand() % n;

  while (begin_position == end_position)
  {
    end_position = rand() % n;
  }

  if (begin_position > end_position)
  {
    swap(begin_position, end_position);
  }

  while (begin_position < end_position)
  {
    swap(
        chromosome[begin_position],
        chromosome[end_position]);

    begin_position++;
    end_position--;
  }
}

void printChromosome()
{
  for (int i = 0; i < NUM_OF_CUSTOMERS; i++)
  {
    cout
        << offspring[0].cromossome[i]
        << " ";
  }

  cout << endl;
}

/*implement your heuristic in this function*/
void run_heuristic()
{

  int parent1, parent2;
  double mutation_rand, crossover_rand;

  parent1 = parent_selection(population);

  parent2 = parent1;

  mutation_rand = rand() / (RAND_MAX * 1.0);
  crossover_rand = rand() / (RAND_MAX * 1.0);

  while (parent2 == parent1)
  {
    parent2 = parent_selection(population);
  }

  if (crossover_rand <= crossover_rate)
  {
    crossover(parent1, parent2);

    if (mutation_rand <= mutation_rate)
    {
      if (mutation_method == "ins")
      {
        mutation_insertion();
      }
      else if (mutation_method == "swp")
      {
        mutation_swap();
      }
      else if (mutation_method == "mutation_mix")
      {
        mutation_mix();
      }
      else
      {
        mutation_inversion();
      }
    }

    take_route(&offspring[0]);

    change_pop();
  }

  get_fitness_mean();

  get_fitness_std();
}

/*free memory structures*/
void free_heuristic()
{
  /*
   * Free the internal arrays of every population individual.
   */
  if (population != nullptr)
  {
    for (int i = 0; i < n_pop; i++)
    {
      delete[] population[i].tour;
      delete[] population[i].cromossome;

      population[i].tour = nullptr;
      population[i].cromossome = nullptr;
    }

    delete[] population;
    population = nullptr;
  }

  /*
   * Only offspring[0] was allocated internally.
   */
  if (offspring != nullptr)
  {
    delete[] offspring[0].tour;
    delete[] offspring[0].cromossome;

    offspring[0].tour = nullptr;
    offspring[0].cromossome = nullptr;

    delete[] offspring;
    offspring = nullptr;
  }

  /*
   * Free every row of the Levenshtein matrix.
   */
  if (lv_distance != nullptr)
  {
    for (int i = 0; i <= NUM_OF_CUSTOMERS; i++)
    {
      delete[] lv_distance[i];
      lv_distance[i] = nullptr;
    }

    delete[] lv_distance;
    lv_distance = nullptr;
  }

  /*
   * best_sol points to an element inside population,
   * so it must not be deleted separately.
   */
  best_sol = nullptr;
}
