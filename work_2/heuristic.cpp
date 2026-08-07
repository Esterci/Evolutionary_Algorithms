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
 * Checks whether a node represents a customer.
 *
 * Assumes:
 *   depot: DEPOT
 *   customers: 1 to NUM_OF_CUSTOMERS
 *   charging stations: above NUM_OF_CUSTOMERS
 */
bool is_customer_node(int node)
{
  return node >= 1 &&
         node <= NUM_OF_CUSTOMERS;
}


/*
 * Checks whether the vehicle can travel from "from"
 * to "to" using the remaining battery charge.
 */
bool can_reach_node(
    int from,
    int to,
    double energy_used)
{
  double energy_required =
      get_energy_consumption(from, to);

  return energy_used + energy_required <=
         BATTERY_CAPACITY + EPS;
}


/*
 * Finds the reachable recharge point that is closest
 * to the next customer.
 *
 * The depot is also considered a recharge point.
 *
 * A recharge point is considered valid when:
 *
 *   1. it can be reached using the remaining battery;
 *   2. after recharging, the next customer can be
 *      reached directly.
 *
 * Returns:
 *   the selected recharge-point index;
 *   -1 when no valid recharge point exists.
 */
int nearest_recharge_to_customer(
    int from,
    int customer,
    double energy_used)
{
  int best_point = -1;

  double best_distance_to_customer =
      std::numeric_limits<double>::infinity();

  double best_distance_from_current =
      std::numeric_limits<double>::infinity();

  /*
   * Check the depot as a recharge point.
   */
  if (from != DEPOT)
  {
    double energy_to_depot =
        get_energy_consumption(
            from,
            DEPOT);

    double depot_to_customer =
        get_energy_consumption(
            DEPOT,
            customer);

    bool depot_is_reachable =
        energy_used + energy_to_depot <=
        BATTERY_CAPACITY + EPS;

    bool customer_is_reachable_from_depot =
        depot_to_customer <=
        BATTERY_CAPACITY + EPS;

    if (depot_is_reachable &&
        customer_is_reachable_from_depot)
    {
      best_point = DEPOT;

      best_distance_to_customer =
          depot_to_customer;

      best_distance_from_current =
          energy_to_depot;
    }
  }

  /*
   * Check all charging stations.
   */
  for (int station = NUM_OF_CUSTOMERS + 1;
       station < ACTUAL_PROBLEM_SIZE;
       station++)
  {
    if (!is_charging_station(station))
    {
      continue;
    }

    if (station == from)
    {
      continue;
    }

    double energy_to_station =
        get_energy_consumption(
            from,
            station);

    /*
     * The station must be reachable using
     * the remaining battery charge.
     */
    if (energy_used + energy_to_station >
        BATTERY_CAPACITY + EPS)
    {
      continue;
    }

    double station_to_customer =
        get_energy_consumption(
            station,
            customer);

    /*
     * After recharging, the vehicle must be able
     * to reach the next customer directly.
     */
    if (station_to_customer >
        BATTERY_CAPACITY + EPS)
    {
      continue;
    }

    /*
     * Select the reachable recharge point closest
     * to the next customer.
     *
     * In case of a tie, select the recharge point
     * closest to the current vehicle position.
     */
    bool closer_to_customer =
        station_to_customer <
        best_distance_to_customer - EPS;

    bool same_customer_distance =
        station_to_customer <=
            best_distance_to_customer + EPS &&
        station_to_customer >=
            best_distance_to_customer - EPS;

    bool closer_to_current =
        energy_to_station <
        best_distance_from_current;

    if (closer_to_customer ||
        (same_customer_distance &&
         closer_to_current))
    {
      best_point = station;

      best_distance_to_customer =
          station_to_customer;

      best_distance_from_current =
          energy_to_station;
    }
  }

  return best_point;
}


/*
 * Finds a reachable charging station that moves
 * the vehicle closer to the depot.
 *
 * This function allows the vehicle to use multiple
 * charging stations before reaching the depot.
 *
 * A charging station is considered valid when:
 *
 *   1. it can be reached using the remaining battery;
 *   2. it is closer to the depot than the current node.
 *
 * Returns:
 *   the selected charging-station index;
 *   -1 when no valid charging station exists.
 */
int nearest_station_toward_depot(
    int from,
    double energy_used)
{
  int best_station = -1;

  double current_distance_to_depot =
      get_energy_consumption(
          from,
          DEPOT);

  double best_distance_to_depot =
      std::numeric_limits<double>::infinity();

  double best_distance_from_current =
      std::numeric_limits<double>::infinity();

  for (int station = NUM_OF_CUSTOMERS + 1;
       station < ACTUAL_PROBLEM_SIZE;
       station++)
  {
    if (!is_charging_station(station))
    {
      continue;
    }

    if (station == from)
    {
      continue;
    }

    double energy_to_station =
        get_energy_consumption(
            from,
            station);

    /*
     * The station must be reachable using
     * the remaining battery charge.
     */
    if (energy_used + energy_to_station >
        BATTERY_CAPACITY + EPS)
    {
      continue;
    }

    double station_to_depot =
        get_energy_consumption(
            station,
            DEPOT);

    /*
     * The selected station must move the vehicle
     * closer to the depot.
     *
     * This condition also helps prevent cycles
     * between charging stations.
     */
    if (station_to_depot >=
        current_distance_to_depot - EPS)
    {
      continue;
    }

    /*
     * Select the reachable station closest
     * to the depot.
     *
     * In case of a tie, select the station
     * closest to the current vehicle position.
     */
    bool closer_to_depot =
        station_to_depot <
        best_distance_to_depot - EPS;

    bool same_depot_distance =
        station_to_depot <=
            best_distance_to_depot + EPS &&
        station_to_depot >=
            best_distance_to_depot - EPS;

    bool closer_to_current =
        energy_to_station <
        best_distance_from_current;

    if (closer_to_depot ||
        (same_depot_distance &&
         closer_to_current))
    {
      best_station = station;

      best_distance_to_depot =
          station_to_depot;

      best_distance_from_current =
          energy_to_station;
    }
  }

  return best_station;
}


/*
 * Recomputes the used vehicle capacity and consumed
 * battery energy using the current route.
 *
 * This function is called after a rollback.
 */
void recompute_route_state(
    const solution *route,
    double &capacity_used,
    double &energy_used)
{
  capacity_used = 0.0;
  energy_used = 0.0;

  for (int step = 1;
       step < route->steps;
       step++)
  {
    int previous =
        route->tour[step - 1];

    int current =
        route->tour[step];

    energy_used +=
        get_energy_consumption(
            previous,
            current);

    /*
     * The depot restores both the vehicle capacity
     * and the battery charge.
     */
    if (current == DEPOT)
    {
      capacity_used = 0.0;
      energy_used = 0.0;
    }
    /*
     * A charging station restores only
     * the battery charge.
     */
    else if (is_charging_station(current))
    {
      energy_used = 0.0;
    }
    /*
     * A customer increases the used vehicle capacity.
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
 * The chromosome index is decremented so that the
 * removed customer can be processed again.
 *
 * Returns:
 *   true  when the rollback is successfully performed;
 *   false when there is no customer available to remove.
 */
bool rollback_previous_customer(
    solution *route,
    int &customer_index,
    double &capacity_used,
    double &energy_used)
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
   * Only a customer can be removed by this function.
   */
  if (!is_customer_node(last_node))
  {
    return false;
  }

  /*
   * Remove the last customer from the route.
   */
  route->steps--;

  /*
   * Process the removed customer again.
   */
  customer_index--;

  /*
   * Restore the capacity and battery state associated
   * with the remaining route.
   */
  recompute_route_state(
      route,
      capacity_used,
      energy_used);

  return true;
}


/*
 * Inserts a recharge point into the route.
 *
 * The battery is fully recharged at both charging
 * stations and the depot.
 *
 * The vehicle capacity is reset only at the depot.
 */
void insert_recharge_point(
    solution *route,
    int recharge_point,
    double &capacity_used,
    double &energy_used)
{
  route->tour[route->steps] =
      recharge_point;

  route->steps++;

  /*
   * The battery is fully recharged.
   */
  energy_used = 0.0;

  /*
   * Visiting the depot also restores
   * the vehicle load capacity.
   */
  if (recharge_point == DEPOT)
  {
    capacity_used = 0.0;
  }
}


/*
 * Builds a route from the customer sequence stored
 * in the chromosome.
 *
 * Route-construction strategy:
 *
 *   1. Try to visit the next customer directly.
 *   2. If the battery is insufficient, select the
 *      reachable recharge point closest to the
 *      next customer.
 *   3. If no recharge point is reachable, remove
 *      the previous customer and force a recharge
 *      before visiting it again.
 *   4. If the vehicle capacity is insufficient,
 *      return to the depot, using multiple charging
 *      stations when necessary.
 *   5. If the route cannot be repaired, penalize
 *      the solution with INT_MAX.
 *
 * This function does not return a boolean value.
 */
void take_route(solution *route)
{
  int customer_index = 0;
  int iterations = 0;

  double capacity_used = 0.0;
  double energy_used = 0.0;

  /*
   * After a rollback, a recharge point must be
   * inserted before trying the removed customer again.
   */
  bool force_recharge = false;

  /*
   * Prevent infinite loops caused by an instance or
   * chromosome that cannot be decoded.
   */
  const int max_iterations =
      20 * (ACTUAL_PROBLEM_SIZE +
            NUM_OF_CUSTOMERS);

  route->steps = 1;
  route->tour_length = INT_MAX;

  route->tour[0] = DEPOT;

  while (true)
  {
    iterations++;

    /*
     * Penalize the solution when the construction
     * exceeds the iteration limit.
     */
    if (iterations > max_iterations)
    {
      route->tour_length = INT_MAX;
      return;
    }

    int from =
        route->tour[route->steps - 1];

    /*
     * All customers have already been inserted.
     * The vehicle must now return to the depot.
     */
    if (customer_index >= NUM_OF_CUSTOMERS)
    {
      /*
       * The route is already closed.
       */
      if (from == DEPOT)
      {
        route->tour_length = 0.0;
        return;
      }

      /*
       * Return directly to the depot when possible.
       */
      if (can_reach_node(
              from,
              DEPOT,
              energy_used))
      {
        route->tour[route->steps] =
            DEPOT;

        route->steps++;

        capacity_used = 0.0;
        energy_used = 0.0;

        route->tour_length = 0.0;
        return;
      }

      /*
       * The depot cannot be reached directly.
       *
       * Insert a reachable charging station that
       * moves the vehicle closer to the depot.
       */
      int charging_station =
          nearest_station_toward_depot(
              from,
              energy_used);

      if (charging_station != -1)
      {
        insert_recharge_point(
            route,
            charging_station,
            capacity_used,
            energy_used);

        /*
         * The next iteration will try to reach
         * the depot again from the new station.
         */
        continue;
      }

      /*
       * No charging station toward the depot
       * can be reached.
       *
       * Remove the last customer and rebuild
       * the final section of the route.
       */
      if (rollback_previous_customer(
              route,
              customer_index,
              capacity_used,
              energy_used))
      {
        force_recharge = true;
        continue;
      }

      /*
       * The route cannot be repaired.
       */
      route->tour_length = INT_MAX;
      return;
    }

    int customer =
        route->cromossome[customer_index];

    double customer_demand =
        get_customer_demand(customer);

    /*
     * A customer whose individual demand exceeds
     * the maximum vehicle capacity cannot be served.
     */
    if (customer_demand >
        MAX_CAPACITY + EPS)
    {
      route->tour_length = INT_MAX;
      return;
    }

    /*
     * After a rollback, insert a recharge point
     * before trying the removed customer again.
     */
    if (force_recharge)
    {
      int recharge_point =
          nearest_recharge_to_customer(
              from,
              customer,
              energy_used);

      if (recharge_point != -1)
      {
        insert_recharge_point(
            route,
            recharge_point,
            capacity_used,
            energy_used);

        force_recharge = false;

        /*
         * The same customer will be processed again
         * after the recharge.
         */
        continue;
      }

      /*
       * One rollback was not sufficient.
       * Remove another customer and try again.
       */
      if (rollback_previous_customer(
              route,
              customer_index,
              capacity_used,
              energy_used))
      {
        force_recharge = true;
        continue;
      }

      /*
       * No additional rollback can be performed.
       */
      route->tour_length = INT_MAX;
      return;
    }

    /*
     * The current vehicle load is insufficient
     * to serve the next customer.
     *
     * The vehicle must return to the depot before
     * processing the same customer again.
     */
    if (capacity_used + customer_demand >
        MAX_CAPACITY + EPS)
    {
      /*
       * Return directly to the depot when possible.
       */
      if (can_reach_node(
              from,
              DEPOT,
              energy_used))
      {
        route->tour[route->steps] =
            DEPOT;

        route->steps++;

        capacity_used = 0.0;
        energy_used = 0.0;

        /*
         * The same customer will be processed again
         * with a restored vehicle load.
         */
        continue;
      }

      /*
       * The depot cannot be reached directly.
       *
       * Move to a reachable charging station that
       * makes progress toward the depot.
       */
      int charging_station =
          nearest_station_toward_depot(
              from,
              energy_used);

      if (charging_station != -1)
      {
        insert_recharge_point(
            route,
            charging_station,
            capacity_used,
            energy_used);

        /*
         * The load is not reset at a charging station.
         * The next iteration will continue trying
         * to reach the depot.
         */
        continue;
      }

      /*
       * No charging station toward the depot
       * can be reached.
       *
       * Remove the previous customer and force
       * a recharge before visiting it again.
       */
      if (rollback_previous_customer(
              route,
              customer_index,
              capacity_used,
              energy_used))
      {
        force_recharge = true;
        continue;
      }

      /*
       * The route cannot be repaired.
       */
      route->tour_length = INT_MAX;
      return;
    }

    /*
     * Visit the next customer directly when there
     * is enough remaining battery charge.
     */
    if (can_reach_node(
            from,
            customer,
            energy_used))
    {
      energy_used +=
          get_energy_consumption(
              from,
              customer);

      capacity_used +=
          customer_demand;

      route->tour[route->steps] =
          customer;

      route->steps++;
      customer_index++;

      continue;
    }

    /*
     * The next customer cannot be reached directly.
     *
     * Select the reachable recharge point closest
     * to that customer.
     */
    int recharge_point =
        nearest_recharge_to_customer(
            from,
            customer,
            energy_used);

    if (recharge_point != -1)
    {
      insert_recharge_point(
          route,
          recharge_point,
          capacity_used,
          energy_used);

      /*
       * The same customer will be processed again
       * after the recharge.
       */
      continue;
    }

    /*
     * No recharge point can be reached.
     *
     * Remove the previous customer and force
     * a recharge before trying it again.
     */
    if (rollback_previous_customer(
            route,
            customer_index,
            capacity_used,
            energy_used))
    {
      force_recharge = true;
      continue;
    }

    /*
     * The route cannot be repaired.
     */
    route->tour_length = INT_MAX;
    return;
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
