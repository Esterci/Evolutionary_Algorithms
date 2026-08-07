from imports import *

problem_instances = [
    "E-n22-k4.evrp",
    "E-n76-k7.evrp",
]

mutation_rates = np.linspace(0.8, 0.95, num=1, endpoint=True, dtype=np.float32)

crossover_rates = np.linspace(0.7, 0.95, num=1, endpoint=True, dtype=np.float32)

mutation_methods = [
    "swp",
    "ins",
    "mix",
    "inv",
]

selection_pressures = np.linspace(1.5, 2.0, num=1, endpoint=True, dtype=np.float32)

population_sizes = np.linspace(2**3, 2**5, num=1, endpoint=True, dtype=np.int16)


def execute_configuration(configuration):
    (
        mutation_rate,
        crossover_rate,
        mutation_method,
        selection_pressure,
        population_size,
        problem_instance,
    ) = configuration

    mutation_rate = float(mutation_rate)
    crossover_rate = float(crossover_rate)
    selection_pressure = float(selection_pressure)
    population_size = int(population_size)

    command = [
        "./main",
        problem_instance,
        str(mutation_rate),
        str(crossover_rate),
        mutation_method,
        str(selection_pressure),
        str(population_size),
        "0",
    ]

    start = time.perf_counter()

    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )

        return {
            "success": True,
            "configuration": configuration,
            "runtime": time.perf_counter() - start,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.CalledProcessError as error:
        return {
            "success": False,
            "configuration": configuration,
            "runtime": time.perf_counter() - start,
            "stdout": error.stdout,
            "stderr": error.stderr,
            "returncode": error.returncode,
        }


configurations = list(
    itertools.product(
        mutation_rates,
        crossover_rates,
        mutation_methods,
        selection_pressures,
        population_sizes,
        problem_instances,
    )
)

# Ajuste conforme a quantidade de núcleos e o consumo de memória do ./main.
max_workers = min(8, os.cpu_count() or 1)

print(f"Total de configurações: {len(configurations)}")
print(f"Execuções simultâneas: {max_workers}")

grid_start = time.perf_counter()

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(execute_configuration, configuration)
        for configuration in configurations
    ]

    for round_number, future in enumerate(as_completed(futures), start=1):
        result = future.result()

        (
            mutation_rate,
            crossover_rate,
            mutation_method,
            selection_pressure,
            population_size,
            problem_instance,
        ) = result["configuration"]

        status = "OK" if result["success"] else "ERRO"

        print(
            f"\n====> Round {round_number}/{len(configurations)} [{status}]"
            f"\nproblem_instance: {problem_instance}"
            f"\nmutation_rate: {float(mutation_rate)}"
            f"\ncrossover_rate: {float(crossover_rate)}"
            f"\nmutation_method: {mutation_method}"
            f"\nselection_pressure: {float(selection_pressure)}"
            f"\nn_pop: {int(population_size)}"
            f"\nreturn code: {result['returncode']}"
            f"\nrun time: {result['runtime']:.2f} segundos"
        )

        if result["stdout"]:
            print("\nSaída:")
            print(result["stdout"])

        if result["stderr"]:
            print("\nErro/diagnóstico:")
            print(result["stderr"])

print(
    f"\nGrid search concluído em " f"{time.perf_counter() - grid_start:.2f} segundos."
)
