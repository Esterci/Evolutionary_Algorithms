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

const double ENERGY_PENALTY = 10.0;
const double CAPACITY_PENALTY = 10.0;
const double REPAIR_PENALTY = 500.0;

const double PENALTY_EXPONENT = 1.0;

bool compare_fitness(const solution &a, const solution &b)
{
  return a.tour_length > b.tour_length;
}

/*
 * Checks whether a node represents a customer.
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
 * Calculates the battery constraint violation.
 *
 * Returns zero when there is no violation.
 */
double energy_violation(
    int from,
    int to,
    double energy_used)
{
  double required_energy =
      energy_used +
      get_energy_consumption(from, to);

  double violation =
      required_energy - BATTERY_CAPACITY;

  if (violation < 0.0)
  {
    violation = 0.0;
  }

  return violation;
}

/*
 * Calculates the vehicle-capacity constraint violation.
 *
 * Returns zero when there is no violation.
 */
double capacity_violation(
    double capacity_used,
    double customer_demand)
{
  double violation =
      capacity_used +
      customer_demand -
      MAX_CAPACITY;

  if (violation < 0.0)
  {
    violation = 0.0;
  }

  return violation;
}

/*
 * Adds a static penalty according to:
 *
 * penalty = C * d^k
 *
 * where:
 *   C = penalty coefficient;
 *   d = constraint violation;
 *   k = penalty exponent.
 */
void add_penalty(
    solution *route,
    double coefficient,
    double violation)
{
  if (violation <= EPS)
  {
    return;
  }

  route->penalty +=
      coefficient *
      pow(violation, PENALTY_EXPONENT);
}

/*
 * Finds the reachable recharge point that is closest
 * to the next customer.
 *
 * The depot is also considered a recharge point.
 *
 * A recharge point is valid when:
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
     * The customer must be reachable after recharging.
     */
    if (station_to_customer >
        BATTERY_CAPACITY + EPS)
    {
      continue;
    }

    bool closer_to_customer =
        station_to_customer <
        best_distance_to_customer - EPS;

    bool same_customer_distance =
        fabs(
            station_to_customer -
            best_distance_to_customer) <= EPS;

    bool closer_to_current =
        energy_to_station <
        best_distance_from_current;

    /*
     * Select the reachable recharge point closest
     * to the next customer.
     *
     * In case of a tie, select the one closest
     * to the current position.
     */
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
 * Multiple charging stations may therefore be used
 * during the return trip to the depot.
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
     * Each selected station must move the vehicle
     * closer to the depot.
     */
    if (station_to_depot >=
        current_distance_to_depot - EPS)
    {
      continue;
    }

    bool closer_to_depot =
        station_to_depot <
        best_distance_to_depot - EPS;

    bool same_depot_distance =
        fabs(
            station_to_depot -
            best_distance_to_depot) <= EPS;

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
 * This function is used after a rollback.
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
     * The depot restores both load capacity
     * and battery charge.
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
     * Visiting a customer increases the used
     * vehicle capacity.
     */
    else if (is_customer_node(current))
    {
      capacity_used +=
          get_customer_demand(current);
    }
  }
}

/*
 * Removes the last customer from the route.
 *
 * The removed customer will be processed again.
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
   * Only customers can be removed.
   */
  if (!is_customer_node(last_node))
  {
    return false;
  }

  route->steps--;

  customer_index--;

  recompute_route_state(
      route,
      capacity_used,
      energy_used);

  return true;
}

/*
 * Inserts a recharge point into the route.
 *
 * Charging stations restore only the battery.
 * The depot restores both battery and load capacity.
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

  energy_used = 0.0;

  if (recharge_point == DEPOT)
  {
    capacity_used = 0.0;
  }
}

/*
 * Finds the smallest battery violation required to
 * reach any recharge point from the current node.
 *
 * The depot is also considered a recharge point.
 *
 * This value represents the distance from the current
 * solution to battery feasibility.
 */
double minimum_recharge_violation(
    int from,
    double energy_used)
{
  double minimum_violation =
      energy_violation(
          from,
          DEPOT,
          energy_used);

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

    double violation =
        energy_violation(
            from,
            station,
            energy_used);

    if (violation < minimum_violation)
    {
      minimum_violation = violation;
    }
  }

  return minimum_violation;
}

/*
 * Builds a route from the chromosome.
 *
 * The heuristic first attempts to maintain feasibility
 * using charging stations, depot returns and rollback.
 *
 * When the route cannot be repaired, the amount of
 * constraint violation is stored as a static penalty
 * instead of assigning INT_MAX.
 */
void take_route(solution *route)
{
  int customer_index = 0;
  int iterations = 0;

  double capacity_used = 0.0;
  double energy_used = 0.0;

  bool force_recharge = false;

  const int max_iterations =
      20 * (ACTUAL_PROBLEM_SIZE +
            NUM_OF_CUSTOMERS);

  route->steps = 1;

  /*
   * No penalty is assigned at the beginning.
   */
  route->penalty = 0.0;

  route->tour_length = 0.0;

  route->tour[0] = DEPOT;

  while (true)
  {
    iterations++;

    /*
     * If the heuristic repeatedly fails to construct
     * the route, add a repair penalty.
     */
    if (iterations > max_iterations)
    {
      add_penalty(
          route,
          REPAIR_PENALTY,
          1.0);

      route->viable = false;
      return;
    }

    int from =
        route->tour[route->steps - 1];

    /*
     * All customers have already been served.
     * The vehicle must now return to the depot.
     */
    if (customer_index >= NUM_OF_CUSTOMERS)
    {
      if (from == DEPOT)
      {
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

        return;
      }

      /*
       * Otherwise, try to return through one or more
       * charging stations.
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

        continue;
      }

      /*
       * If no station is reachable, perform rollback.
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
       *
       * Penalize according to the minimum additional
       * battery energy required to reach a recharge point.
       */
      double violation =
          minimum_recharge_violation(
              from,
              energy_used);

      add_penalty(
          route,
          ENERGY_PENALTY,
          violation);

      route->viable = false;
      return;
    }

    int customer =
        route->cromossome[customer_index];

    double customer_demand =
        get_customer_demand(customer);

    /*
     * Penalize a customer whose individual demand
     * exceeds the maximum vehicle capacity.
     */
    if (customer_demand >
        MAX_CAPACITY + EPS)
    {
      double violation =
          customer_demand -
          MAX_CAPACITY;

      add_penalty(
          route,
          CAPACITY_PENALTY,
          violation);

      route->viable = false;
      return;
    }

    /*
     * After a rollback, a recharge point must be
     * inserted before trying the customer again.
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

        continue;
      }

      /*
       * If one rollback is not enough,
       * continue moving backward.
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
       * No further repair is possible.
       */
      double violation =
          minimum_recharge_violation(
              from,
              energy_used);

      add_penalty(
          route,
          ENERGY_PENALTY,
          violation);
      route->viable = false;
      return;
    }

    /*
     * Check the vehicle load capacity.
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

        continue;
      }

      /*
       * If the depot cannot be reached directly,
       * return through charging stations.
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

        continue;
      }

      /*
       * No route toward the depot is available.
       * Try rollback before applying a penalty.
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
       * The load-capacity violation is proportional
       * to the amount by which capacity is exceeded.
       */
      double violation =
          capacity_violation(
              capacity_used,
              customer_demand);

      add_penalty(
          route,
          CAPACITY_PENALTY,
          violation);
      route->viable = false;
      return;
    }

    /*
     * Visit the customer directly when the remaining
     * battery charge is sufficient.
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
     * The customer cannot be reached directly.
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

      continue;
    }

    /*
     * If no recharge point can be reached,
     * perform rollback.
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
     * No repair is possible.
     *
     * Penalize according to the amount of battery
     * capacity required beyond the available limit.
     */
    double violation =
        energy_violation(
            from,
            customer,
            energy_used);

    add_penalty(
        route,
        ENERGY_PENALTY,
        violation);
    route->viable = false;
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
    population[i].viable = true;

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
    offspring[i].viable = true;
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

  offspring[0].viable = true;

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
  destination.viable = source.viable;
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
