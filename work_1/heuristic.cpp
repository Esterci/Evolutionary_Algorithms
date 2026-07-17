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

/*initialize the structure of your heuristic in this function*/
void initialize_heuristic()
{

  best_sol = new solution;
  best_sol->tour = new int[NUM_OF_CUSTOMERS + 1000];
  best_sol->id = 1;
  best_sol->steps = 0;
  best_sol->tour_length = INT_MAX;
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
  int cociente, resto, bin_stack = NUM_OF_CUSTOMERS * n_bits;

  for (int i = NUM_OF_CUSTOMERS; i >= 1; i--)
  {
    cociente = r[i];

    while (cociente >= 1)
    {

      resto = cociente % 2;

      r_bin[bin_stack] = bool(resto);

      bin_stack--;

      cociente = cociente / 2;
    }
  }
}

/*implement your heuristic in this function*/
void run_heuristic()
{

  /*generate a random solution for the random heuristic*/
  int i, help, object, n_bit, tot_assigned = 0;
  int *r;
  bool *r_bin;
  int from, to;

  n_bit = count_bits(); // Count number of bits to represent costumers id's

  r = new int[NUM_OF_CUSTOMERS + 1];

  r_bin = new bool[(NUM_OF_CUSTOMERS + 1) * n_bit];

  // set indexes of objects
  for (i = 1; i <= NUM_OF_CUSTOMERS; i++)
  {
    r[i - 1] = i;
  }

  // initilize a false bool vector
  for (i = 0; i < (NUM_OF_CUSTOMERS + 1) * n_bit; i++)
  {
    r_bin[i] = 0;
  }

  // randomly change indexes of objects
  for (i = 0; i <= NUM_OF_CUSTOMERS; i++) // Tem como fixarmos e salvarmos a seed?
  {
    object = (int)((rand() / (RAND_MAX + 1.0)) * (double)(NUM_OF_CUSTOMERS - tot_assigned));
    help = r[i];
    r[i] = r[i + object];
    r[i + object] = help;
    tot_assigned++;
  }

  conv_int_bin(r, r_bin, n_bit);

  best_sol->steps = 0;
  best_sol->tour_length = INT_MAX;

  best_sol->tour[0] = DEPOT;
  best_sol->steps++;

  i = 0;

  while (i < NUM_OF_CUSTOMERS)
  {
    from = best_sol->tour[best_sol->steps - 1];
    to = r[i];

    best_sol->tour[best_sol->steps] = to;
    best_sol->steps++;
    i++;
  }

  // close EVRP tour to return back to the depot
  if (best_sol->tour[best_sol->steps - 1] != DEPOT)
  {
    best_sol->tour[best_sol->steps] = DEPOT;
    best_sol->steps++;
  }

  best_sol->tour_length = fitness_evaluation(best_sol->tour, best_sol->steps);

  // free memory
  delete[] r;
  delete[] r_bin;
}

/*free memory structures*/
void free_heuristic()
{

  delete[] best_sol->tour;
}
