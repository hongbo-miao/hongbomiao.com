import logging
import math
from random import SystemRandom

POPULATION_COUNT = 100
MUTATION_RATE = 0.01
GENERATION_COUNT = 100
CHROMOSOME_LENGTH = 16  # Encode x using 16-bit binary
X_MIN = 0
X_MAX = 1
SURVIVOR_PROPORTION = 0.2

logger = logging.getLogger(__name__)

random_number_generator = SystemRandom()


def decode_chromosome_to_value(chromosome: str) -> float:
    # Decode a binary string into a real number x
    value = int(chromosome, 2)
    return X_MIN + (X_MAX - X_MIN) * value / (2**CHROMOSOME_LENGTH - 1)


def calculate_fitness(chromosome: str) -> float:
    # Compute fitness f(x) = x * sin(10πx) + 1
    x = decode_chromosome_to_value(chromosome)
    return x * math.sin(10 * math.pi * x) + 1


def generate_random_chromosome() -> str:
    # Generate a random binary chromosome
    return "".join(
        random_number_generator.choice("01") for _ in range(CHROMOSOME_LENGTH)
    )


def perform_single_point_crossover(
    parent_chromosome_one: str,
    parent_chromosome_two: str,
) -> str:
    point = random_number_generator.randint(1, CHROMOSOME_LENGTH - 1)
    return parent_chromosome_one[:point] + parent_chromosome_two[point:]


def mutate_chromosome(chromosome: str) -> str:
    # Bit mutation
    bits = list(chromosome)
    for i in range(len(bits)):
        if random_number_generator.random() < MUTATION_RATE:
            bits[i] = "1" if bits[i] == "0" else "0"
    return "".join(bits)


def main() -> None:
    population = [generate_random_chromosome() for _ in range(POPULATION_COUNT)]

    for generation_number in range(GENERATION_COUNT):
        # Calculate fitness
        scored_population = [
            (chromosome, calculate_fitness(chromosome)) for chromosome in population
        ]
        scored_population.sort(key=lambda x: x[1], reverse=True)
        best_chromosome, best_fitness = scored_population[0]
        best_x = decode_chromosome_to_value(best_chromosome)

        logger.info(
            f"Generation {generation_number}: x = {best_x:.5f}, f(x) = {best_fitness:.5f}",
        )

        # Selection (elitism + random)
        survivor_count = max(2, math.ceil(POPULATION_COUNT * SURVIVOR_PROPORTION))
        survivors = [
            chromosome for chromosome, _fitness in scored_population[:survivor_count]
        ]

        # Reproduction
        new_population: list[str] = survivors[:]
        while len(new_population) < POPULATION_COUNT:
            parent_chromosome_one, parent_chromosome_two = (
                random_number_generator.sample(survivors, 2)
            )
            child_chromosome = perform_single_point_crossover(
                parent_chromosome_one,
                parent_chromosome_two,
            )
            child_chromosome = mutate_chromosome(child_chromosome)
            new_population.append(child_chromosome)

        population = new_population


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
