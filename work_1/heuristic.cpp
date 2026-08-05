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

bool compare_fitness(const solution &a, const solution &b)
{
  return a.tour_length > b.tour_length;
}

void take_route(solution *route)
{
  /*generate a random solution for the random heuristic*/
  int i;
  int to;

  route->steps = 1;
  route->tour_length = INT_MAX;

  route->tour[0] = DEPOT;

  i = 0;
  while (i < NUM_OF_CUSTOMERS)
  {
    to = route->cromossome[i];

    route->tour[route->steps] = to;
    route->steps++;
    i++;
  }

  // close EVRP tour to return back to the depot
  if (route->tour[route->steps - 1] != DEPOT)
  {
    route->tour[route->steps] = DEPOT;
    route->steps++;
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
    for (j = 0; j <= NUM_OF_CUSTOMERS; j++) // Tem como fixarmos e salvarmos a seed?
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

    if(mutation_rand <= mutation_rate){
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

    change_pop();
  }
  
  take_route(&offspring[0]);

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
