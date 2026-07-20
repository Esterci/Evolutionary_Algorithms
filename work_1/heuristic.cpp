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
int n_bit = 0;

void take_route(solution *route)
{
  int i, j;
  double best_fitness = INT_MAX;
  int from, to;

  for (i = 0; i < n_pop; i++)
  {
    population[i].steps = 1;
    population[i].tour[0] = DEPOT;

    j = 0;

    while (j < NUM_OF_CUSTOMERS)
    {
      from = route[i].tour[route[i].steps - 1];
      to = route[i].real_cromossome[j];

      route[i].tour[route[i].steps] = to;
      route[i].steps++;
      j++;
    }

    // close EVRP tour to return back to the depot
    if (route[i].tour[route[i].steps - 1] != DEPOT)
    {
      route[i].tour[route[i].steps] = DEPOT;
      route[i].steps++;
    }

    route[i].tour_length = fitness_evaluation(route[i].tour, route[i].steps);

    if (route[i].tour_length < best_fitness)
    {
      best_sol = &route[i];
    }
  }
}

/*initialize the structure of your heuristic in this function*/
void initialize_heuristic(int run)
{
  /*generate a random solution for the random heuristic*/
  int i, j, help, object, tot_assigned;

  n_bit = count_bits(); // Count number of bits to represent costumers id's

  // Aloca um vetor com n_pop soluções
  population = new solution[n_pop];
  offspring = new solution[n_pop];

  for (i = 0; i < n_pop; i++)
  {

    population[i].tour = new int[NUM_OF_CUSTOMERS + 1000];
    population[i].real_cromossome = new int[NUM_OF_CUSTOMERS + 1];
    population[i].cromossome = new bool[(NUM_OF_CUSTOMERS + 1) * n_bit];
    population[i].id = i + 1;
    population[i].steps = 1;
    population[i].tour_length = INT_MAX;

    offspring[i].tour = new int[NUM_OF_CUSTOMERS + 1000];
    offspring[i].real_cromossome = new int[NUM_OF_CUSTOMERS + 1];
    offspring[i].cromossome = new bool[(NUM_OF_CUSTOMERS + 1) * n_bit];
    offspring[i].id = i + 1;
    offspring[i].steps = 1;
    offspring[i].tour_length = INT_MAX;

    // initilize a false bool vector
    for (j = 0; j < (NUM_OF_CUSTOMERS + 1) * n_bit; j++)
    {
      population[i].cromossome[j] = false;
      offspring[i].cromossome[j] = false;
    }
  }

  // Inicialmente, a primeira solução é considerada a melhor
  best_sol = &population[0];

  cout << "Run: " << run << endl;

  for (i = 0; i < n_pop; i++)
  {
    cout << "   -> seed: " << (run - 1) * (n_pop) + i + 1 << endl;

    srand((run - 1) * n_pop + i + 1); // random seed

    tot_assigned = 0;

    for (j = 0; j < NUM_OF_CUSTOMERS; j++)
    {
      population[i].real_cromossome[j] = j + 1;
    }

    // randomly change indexes of obiects
    for (j = 0; j <= NUM_OF_CUSTOMERS; j++) // Tem como fixarmos e salvarmos a seed?
    {
      object = (int)((rand() / (RAND_MAX + 1.0)) * (double)(NUM_OF_CUSTOMERS - tot_assigned));
      help = population[i].real_cromossome[j];
      population[i].real_cromossome[j] = population[i].real_cromossome[j + object];
      population[i].real_cromossome[j + object] = help;
      tot_assigned++;
    }

    conv_int_bin(population[i].real_cromossome, population[i].cromossome);
  }

  take_route(population);
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

void conv_int_bin(int *r, bool *r_bin)
{
  int bin_stack = NUM_OF_CUSTOMERS * n_bit;

  for (int i = NUM_OF_CUSTOMERS - 1; i >= 0; i--)
  {
    int valor = r[i];

    for (int bit = 0; bit < n_bit; bit++)
    {
      int valor_bit = valor % 2;

      r_bin[bin_stack] = bool(valor_bit);

      valor = valor / 2;
      bin_stack--;
    }
  }
}

void crossover()
{
  int cross_point, customer_cut, n_gens = NUM_OF_CUSTOMERS * n_bit;

  for (int parent = 0; parent < n_pop - 1; parent += 2)
  {
    customer_cut = rand() % (NUM_OF_CUSTOMERS);

    cross_point = customer_cut * n_bit;

    cout << customer_cut << ", " << cross_point << endl;

    for (int j = 0; j < n_gens; j++)
    {
      if (j < cross_point)
      {
        offspring[parent].cromossome[j] =
            population[parent].cromossome[j];

        offspring[parent + 1].cromossome[j] =
            population[parent + 1].cromossome[j];
      }
      else
      {
        offspring[parent].cromossome[j] =
            population[parent + 1].cromossome[j];

        offspring[parent + 1].cromossome[j] =
            population[parent].cromossome[j];
      }
    }
  }
}

void change_pop()
{
  int soma, j, cromossome_len = NUM_OF_CUSTOMERS * n_bit;
  bool valido, valor_bit;

  for (int i = 0; i < n_pop; i++)
  {

    valido = true;

    for (int gene = 0; gene < NUM_OF_CUSTOMERS; gene++)
    {

      soma = 0;

      for (int bit = 0; bit < n_bit; bit++)
      {
        int posicao =
            gene * n_bit + bit;

        valor_bit = offspring[i].cromossome[posicao];

        soma += valor_bit * pow(2, (n_bit - bit - 1));
      }

      if (soma < 1 || soma > NUM_OF_CUSTOMERS)
      {
        valido = false;
        break;
      }
    }

    if (valido)
    {
      cout << "V valido" << endl;
      for (j = 0; j < cromossome_len; j++)
      {
        population[i].cromossome[j] = offspring[i].cromossome[j];
      }

      for (j = 0; j < NUM_OF_CUSTOMERS; j++)
      {
        population[i].real_cromossome[j] = offspring[i].real_cromossome[j];
      }
    }
    else
    {
      cout << "X invalido" << endl;

      for (j = 0; j < cromossome_len; j++)
      {
        cout << offspring[i].cromossome[j];
      }

      cout << endl;
    }
  }
}

/*implement your heuristic in this function*/
void run_heuristic()
{

  crossover();

  change_pop();

  take_route(population);
}

/*free memory structures*/
void free_heuristic()
{

  delete[] best_sol->tour;
  delete[] population;
  delete[] offspring;
}
