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

#include "heuristic.hpp"
#include "EVRP.hpp"

using namespace std;

solution *best_sol; // see heuristic.hpp for the solution structure
solution *population;
solution *offspring;

// GA parameters
int n_pop = 8;
int n_offspring = 1;

bool compare_fitness(const solution &a, const solution &b)
{
  return a.tour_length < b.tour_length;
}

void take_route(solution *route)
{
  /*generate a random solution for the random heuristic*/
  int i, help, object, tot_assigned = 0;
  int to;

  // set indexes of objects
  for (i = 1; i <= NUM_OF_CUSTOMERS; i++)
  {
    route->cromossome[i - 1] = i;
  }
  // randomly change indexes of objects
  for (i = 0; i <= NUM_OF_CUSTOMERS; i++)
  {
    object = (int)((rand() / (RAND_MAX + 1.0)) * (double)(NUM_OF_CUSTOMERS - tot_assigned));
    help = route->cromossome[i];
    route->cromossome[i] = route->cromossome[i + object];
    route->cromossome[i + object] = help;
    tot_assigned++;
  }

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

/*initialize the structure of your heuristic in this function*/
void initialize_heuristic(int run)
{
  /*generate a random solution for the random heuristic*/
  int i, j, help, object, tot_assigned;

  // Aloca um vetor com n_pop soluções
  population = new solution[n_pop];
  offspring = new solution[n_pop];

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

  for (i = 0; i < n_offspring; i++)
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

  int totalWeight = 0;

  // O melhor recebe o maior peso.
  for (int i = 0; i < n_pop; i++)
  {
    ranked[i].weight = n_pop - i;
    totalWeight += ranked[i].weight;
  }

  // Sorteia entre 1 e totalWeight.
  int randomValue = rand() % totalWeight + 1;

  int accumulatedWeight = 0;

  for (int i = 0; i < n_pop; i++)
  {
    accumulatedWeight += ranked[i].weight;

    if (randomValue <= accumulatedWeight)
    {
      return i;
    }
  }

  return NUM_OF_CUSTOMERS - 1;
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

  take_route(&offspring[0]);
}

void change_pop()
{
  // Test fitness of offspring
  offspring[0].tour_length = fitness_evaluation(offspring[0].tour, offspring[0].steps);

  if (offspring[0].tour_length < best_sol->tour_length)
  {
    best_sol = &offspring[0];
  }

  // The survival of the fittest
  // replace less fit individual by the offspring
  for (int i = 0; i < NUM_OF_CUSTOMERS; i++)
  {
    population[n_pop - 1].cromossome[i] = offspring[0].cromossome[i];
  }

  for (int i = 0; i < NUM_OF_CUSTOMERS; i++)
  {
    population[n_pop - 1].cromossome[i] = offspring[0].cromossome[i];
  }

  population[n_pop - 1].tour_length = offspring[0].tour_length;
}

/*implement your heuristic in this function*/
void run_heuristic()
{
  int parent1, parent2 = 0;

  parent1 = parent_selection(population);

  while (parent2 == parent1)
  {
    parent2 = parent_selection(population);
  }

  crossover(parent1, parent2);

  change_pop();
}

/*free memory structures*/
void free_heuristic()
{

  delete[] best_sol->tour;
  delete[] population;
  delete[] offspring;
}
