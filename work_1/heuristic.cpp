#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <math.h>
#include <fstream>
#include <limits.h>

#include "heuristic.hpp"
#include "EVRP.hpp"

using namespace std;

solution *best_sol; // see heuristic.hpp for the solution structure
solution *population;
solution *offspring;

// GA parameters
int n_pop = 4;

/*initialize the structure of your heuristic in this function*/
void initialize_heuristic()
{
  best_sol = new solution;

  // Aloca um vetor com n_pop soluções
  population = new solution[n_pop];
  offspring = new solution[n_pop];

  for (int i = 0; i < n_pop; i++)
  {
    population[i].tour = new int[NUM_OF_CUSTOMERS + 1000];
    population[i].id = i + 1;
    population[i].steps = 0;
    population[i].tour_length = INT_MAX;

    offspring[i].tour = new int[NUM_OF_CUSTOMERS + 1000];
    offspring[i].id = i + 1;
    offspring[i].steps = 0;
    offspring[i].tour_length = INT_MAX;
  }

  // Inicialmente, a primeira solução é considerada a melhor
  best_sol = &population[0];
}

int count_bits()
{

  int cociente = NUM_OF_CUSTOMERS, cont = 1;

  while (cociente >= 2)
  {

    cociente = cociente / 2;
    cont++;
  }

  return cont;
}

void conv_int_bin(int *r, bool *r_bin, int n_bits)
{
  int bin_stack = NUM_OF_CUSTOMERS * n_bits;

  for (int i = NUM_OF_CUSTOMERS - 1; i >= 0; i--)
  {
    int valor = r[i];

    for (int bit = 0; bit < n_bits; bit++)
    {
      int valor_bit = valor % 2;

      r_bin[bin_stack] = bool(valor_bit);

      cout << r_bin[bin_stack] << "-";

      valor = valor / 2;
      bin_stack--;
    }
  }
}

void crossover(int n_bit)
{
  int n_gens = (NUM_OF_CUSTOMERS)*n_bit;
  int cross_point = rand() / (RAND_MAX + 1.0) * n_gens;

  // cout << "Começando crossover" << endl;

  for (int parent = 0; parent < n_pop - 1; parent += 2)
  {
    // cout << "parents: " << parent << "," << parent + 1 << endl;
    // cout << "crossover point: " << cross_point << endl;

    for (int j = 0; j < n_gens; j++)
    {
      if (j < cross_point)
      {
        offspring[parent].cromossome[j] = population[parent].cromossome[j];
        offspring[parent + 1].cromossome[j] = population[parent + 1].cromossome[j];
      }
      else
      {
        offspring[parent].cromossome[j] = population[parent + 1].cromossome[j];
        offspring[parent + 1].cromossome[j] = population[parent].cromossome[j];
      }

      // cout << "p1g" << j << ":" << population[parent].cromossome[j] << ",";
      // cout << "p2g" << j << ":" << population[parent + 1].cromossome[j] << endl
      //  << endl;
      // cout << "o1g" << j << ":" << population[parent].cromossome[j] << ",";
      // cout << "o2g" << j << ":" << population[parent + 1].cromossome[j] << endl;
    }
    // cout << endl;
    population[parent] = offspring[parent];
    population[parent + 1] = offspring[parent + 1];
  }
}

/*implement your heuristic in this function*/
void run_heuristic(int run)
{

  cout << "Run: " << run << endl;

  /*generate a random solution for the random heuristic*/
  int i, help, object, n_bit, tot_assigned;
  int *r;
  bool *r_bin;
  double best_fitness = INT_MAX;
  int from, to;

  n_bit = count_bits(); // Count number of bits to represent costumers id's

  r = new int[NUM_OF_CUSTOMERS + 1];

  r_bin = new bool[(NUM_OF_CUSTOMERS + 1) * n_bit];

  for (int j = 0; j < n_pop; j++)
  {
    cout << "seed: " << (run - 1) * (n_pop) + j + 1 << endl;

    srand((run - 1) * n_pop + j + 1); // random seed

    tot_assigned = 0;

    for (i = 0; i < NUM_OF_CUSTOMERS; i++)
    {
      r[i] = i + 1;
    }

    // initilize a false bool vector
    for (i = 0; i < (NUM_OF_CUSTOMERS + 1) * n_bit; i++)
    {
      r_bin[i] = bool(0);
    }

    offspring[j].cromossome = r_bin;

    cout << "parent cromossome: ";
    // randomly change indexes of objects
    for (i = 0; i <= NUM_OF_CUSTOMERS; i++) // Tem como fixarmos e salvarmos a seed?
    {
      object = (int)((rand() / (RAND_MAX + 1.0)) * (double)(NUM_OF_CUSTOMERS - tot_assigned));
      help = r[i];
      r[i] = r[i + object];
      r[i + object] = help;
      tot_assigned++;

      cout << r[i] << "-";
    }
    cout << endl;

    cout << "parent bin cromossome: ";
    conv_int_bin(r, r_bin, n_bit);
    cout << endl;

    population[j].cromossome = r_bin;

    population[j].steps = 0;
    population[j].tour_length = INT_MAX;

    population[j].tour[0] = DEPOT;
    population[j].steps++;
  }

  crossover(n_bit);

  for (int j = 0; j < n_pop; j++)
  {
    i = 0;

    while (i < NUM_OF_CUSTOMERS)
    {
      from = population[j].tour[population[j].steps - 1];
      to = r[i];

      population[j].tour[population[j].steps] = to;
      population[j].steps++;
      i++;
    }

    // close EVRP tour to return back to the depot
    if (population[j].tour[population[j].steps - 1] != DEPOT)
    {
      population[j].tour[population[j].steps] = DEPOT;
      population[j].steps++;
    }

    population[j].tour_length = fitness_evaluation(population[j].tour, population[j].steps);

    if (population[j].tour_length < best_fitness)
    {
      best_sol = &population[j];
    }

    cout << endl
         << endl;
  }

  // free memory
  delete[] r;
  delete[] r_bin;
}

/*free memory structures*/
void free_heuristic()
{

  delete[] best_sol->tour;
  delete[] population;
}
