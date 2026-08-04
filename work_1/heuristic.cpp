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
#include <random>
#include <cstdlib>
#include <ctime>
#include "heuristic.hpp"
#include "EVRP.hpp"

using namespace std;

solution *best_sol; // see heuristic.hpp for the solution structure
solution *population;
solution *offspring;

int n_pop;
float mutation_rate;
float crossover_rate;
char *mutation_method;
float select_pres;

bool compare_fitness(const solution &a, const solution &b)
{
  return a.tour_length > b.tour_length;
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

double linear_classification(int i)
{
  return (2 - select_pres) / n_pop + 2 * i * (select_pres - 1) / (n_pop * (n_pop - 1));
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
    population[0].cromossome[i] = offspring[0].cromossome[i];
  }

  for (int i = 0; i < NUM_OF_CUSTOMERS; i++)
  {
    population[0].cromossome[i] = offspring[0].cromossome[i];
  }

  population[0].tour_length = offspring[0].tour_length;
}

/*
Mutação por Inserção:

int n;
int origin;
int destination;
int gene;
int i;

void mutation_insertion()
{
  n = tamanho do cromossomo

  if (n < 2)
  {
    return;
  }
// Seleciona aleatoriamento posição do gene;

  origin = rand () % n

// Seleciona nova posição;

destination = rand () % n;

// Posições precisam ser diferentes;

while (destination == origem)
{
  destination = rand() % n;
}
  // Guarda gene sorteado;
  gene = chromossome[origin];

  if (origin < destination)  // Desloca para esquerda;
  {
    for (i = origin, i < destination; i++)
  {
    chromossome[i]=chromossome[i+1];
    }
  }
  else       // desloca para direita;
  {
    for (i = origin; i>destination; i--)
    {
      chromossome[i] = chromossome[i-1];
    }
  }
    // Insere gene na nova posição sorteada;
      chromossome[destination] = gene;
}

*/

/*
// Mutação por mistura:

// Variáveis globais
std::vector<int> chromosome = {};
std::vector<int> positions;

int n;    // armarzena o tamanho do cromossomo.
int quantity; // Armazena quantas posições serão escolhidas para a mistura.
int position; // Guarda temporariamente uma posição sorteada.
int i;        
int j;
int temp;

bool repeated;  //se uma posição sorteada já está no vetor positions.


void mutation_mix()
{
    n = chromosome.size();

    if (n < 2)
    {
        return;
    }

    positions.clear();      // Limpa as posições de uma mutação anterior;

    // Seleciona aleatoriamente quantos genes serão misturados - 
    quantity = 2 + rand() % (n - 1);    // Seleciona 2 até 6 gene (?) conferir;

    // Seleciona posições aleatórias
    while (positions.size() < quantity)
    {
        position = rand() % n;  //sorteia a posição;

        repeated = false;

        // Verifica se a posição já foi selecionada

        for (i = 0; i < positions.size(); i++)
        {
            if (positions[i] == position)
            {
                repeated = true;
            }
        }

        // Guarda somente posições diferentes
        if (repeated == false)
        {
            positions.push_back(position); //
        }
    }

    // Mistura os genes das posições selecionadas

    for (i = quantity - 1; i > 0; i--)
    {
        // Seleciona outra posição dentro do conjunto
        j = rand() % i;

        // Troca os genes
        temp = chromosome[positions[i]];

        chromosome[positions[i]] = chromosome[positions[j]];  //Copiando o segundo gene;

        chromosome[positions[j]] = temp;  //Finalizando a troca;
    }
}

void printChromosome()
{
    for (i = 0; i < chromosome.size(); i++)
    {
        std::cout << chromosome[i] << " ";
    }

    std::cout << std::endl;
}

int main()  // Para todas as mutações;
{
    srand(time(NULL)); // valor atual;

    std::cout << "Antes da mutacao:" << std::endl;
    printChromosome();

    mutation_mix();

    std::cout << "\nPosicoes selecionadas: ";

    for (i = 0; i < positions.size(); i++)
    {
        std::cout << positions[i] << " ";
    }

    std::cout << "\n\nDepois da mutacao:" << std::endl;
    printChromosome();

    return 0;
}

*/

/*
// Mutação por inversão:

// Variáveis globais
int n;
int begin_position;
int end_position;
int temp;
int i;

void mutation_inversion()
{
    // Obtém o tamanho do cromossomo
    n = chromosome.size();

    // É necessário ter pelo menos dois genes
    if (n < 2)
    {
        return;
    }

    // Seleciona aleatoriamente o início do intervalo
    begin_position = rand() % n;

    // Seleciona aleatoriamente o fim do intervalo
    end_position = rand() % n;

    // As posições precisam ser diferentes
    while (begin_position == end_position)
    {
        end_position = rand() % n;
    }

  
    Se a posição inicial for maior que a final,
    troca as duas posições.
  
    if (begin_position > end_position)
    {
        temp = begin_position;
        begin_position = end_position;
        end_position = temp;
    }

  
    Inverte os genes dentro do intervalo.

    O gene do início é trocado com o gene do fim.
    Depois, as duas posições se aproximam.
  
    while (begin_position < end_position)
    {
        temp = chromosome[begin_position];

        chromosome[begin_position] =
            chromosome[end_position];

        chromosome[end_position] = temp;

        begin_position++;
        end_position--;
    }
}

void printChromosome()
{
    for (i = 0; i < chromosome.size(); i++)
    {
        std::cout << chromosome[i] << " ";
    }

    std::cout << std::endl;
}

int main()
{
    // Inicializa os números aleatórios
    srand(time(NULL));

    std::cout << "Antes da mutacao:" << std::endl;
    printChromosome();

    mutation_inversion();

    std::cout << "\nDepois da mutacao:" << std::endl;
    printChromosome();

    return 0;
}


*/


void mutation()
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

/*implement your heuristic in this function*/
void run_heuristic()
{

  int parent1, parent2;

  parent1 = parent_selection(population);

  parent2 = parent1;

  while (parent2 == parent1)
  {
    parent2 = parent_selection(population);
  }
  
  crossover(parent1, parent2);

  mutation();

  change_pop();
}

/*free memory structures*/
void free_heuristic()
{

  delete[] best_sol->tour;
  delete[] population;
  delete[] offspring;
}
