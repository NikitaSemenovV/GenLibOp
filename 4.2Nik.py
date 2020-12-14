# Количество особей всегда 200
# Начальная популяция - жадный выбор, начиная со случайного груза (1.2)
# Скрещивание - однородный (каждый бит от случайно выбранного родителя) (3.2)
# Отбор - выбрать только 20% самых приспособленных особей (2.2)
# Новая популяция - замена своих родителей (5.3)
# Мутация - случайное изменение 3х битов у 5% особей (4.2)

import random as rnd
from random import randint

import numpy as np
from numpy import loadtxt
from numpy.random import randint

weight, value, capacity = loadtxt("23.txt", dtype='float', comments="#", delimiter=" ", skiprows=1, unpack=True)

file = open('23.txt', 'r')
first_str = file.readline().split(" ")
file.close()
knapsack_0 = int(first_str[0].strip())  # Макс вес который может выдержать сумка
knapsack_1 = int(first_str[1].strip())  # макс емкость

num_items = 200

item_list = np.zeros((len(weight), 3))
for i in range(len(weight)):
    item_list[i, 0] = weight[i]
    item_list[i, 1] = value[i]
    item_list[i, 2] = capacity[i]
item_list.sort()


# Создание изначальной популяции: жадный выбор начиная со случайного груза
def individual(number_of_genes):
    global item_list
    indiv = np.zeros(number_of_genes)
    ind = rnd.randint(0, number_of_genes - 1)
    w, v = 0, 0
    for i in range(number_of_genes):
        if i < ind or w >= knapsack_0 or v >= knapsack_1:
            indiv[i] = 0.0
        else:
            w += item_list[i, 2]
            v += item_list[i, 0]
            if w < knapsack_0 and v < knapsack_1:
                indiv[i] = 1.0
    return indiv


def population(number_of_individuals,
               number_of_genes):
    return [individual(number_of_genes)
            for x in range(number_of_individuals)]


# Фитнесс-функция
def fitness_calculation(individual):
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    for i in range(len(individual)):
        a1 += individual[i] * value[i]
        a2 += individual[i] * weight[i]
        a3 += individual[i] * capacity[i]
    if a2 <= knapsack_0 and a1 <= knapsack_1:
        fitness = a3
    else:
        fitness = 0
    return fitness


# Функция отбора особей для скрещивания: выбрать только 20% самых приспособленных особей
def selection(generation):
    generation['Normalized Fitness'] = sorted([generation['Fitness'][x] / sum(generation['Fitness'])
                                               for x in range(len(generation['Fitness']))], reverse=True)
    generation['Cumulative Sum'] = np.array(generation['Normalized Fitness']).cumsum()

    select_individuals = [generation['Individuals'][-x - 1]
                          for x in range(int(len(generation['Individuals']) // 5))]
    select_fitnesses = [generation['Fitness'][-x - 1]
                        for x in range(int(len(generation['Individuals']) // 5))]
    select = {'Individuals': select_individuals,
              'Fitness': select_fitnesses}
    return select


# Функция рандомного выбора родителей(создание пар)
def pairing(selec):
    individuals = selec['Individuals']
    fitness = selec['Fitness']
    parents = []
    for x in range(len(individuals) // 2):
        parents.append(
            [individuals[randint(0, (len(individuals) - 1))], individuals[randint(0, (len(individuals) - 1))]])
        cond = parents[x][0] == parents[x][1]
        if type(cond) != type(True):
            cond = (parents[x][0] == parents[x][1]).all()
        while cond:
            parents[x][1] = individuals[randint(0, (len(individuals) - 1))]
            cond = parents[x][0] == parents[x][1]
            if type(cond) != type(True):
                cond = (parents[x][0] == parents[x][1]).all()
    return parents


# Скрещивание между выбранными особями. Каждая особь скрещивается 1 раз за 1 поколение, 1 пара дает 2 потомка
def mating(parents):
    offspr1 = [0 for i in range(len(parents[0]))]
    offspr2 = [0 for i in range(len(parents[0]))]
    for i in range(len(parents[0])):
        offspr1[i] = parents[rnd.randint(0, 1)][i]
        offspr2[i] = parents[rnd.randint(0, 1)][i]
    return [offspr1, offspr2]


# Мутация :4.2 случайное изменение 3х битов у 5% особей

def mutation(unmutated):
    num_mutate = round(len(unmutated) * 0.05)
    for x in range(num_mutate):
        n = rnd.randint(0, len(unmutated) - 1)
        individual = unmutated[n]
        for y in range(3):
            individual[rnd.randint(0, len(individual) - 1)] = rnd.randint(0, 1)
        unmutated[n] = individual
    with_mutation = unmutated
    return with_mutation


# Формирование новой популяции 5.3 замена своих родителей
def eqGen(arr, gen):
    for i in range(len(arr)):
        cond = arr[i] == gen
        if type(cond) != type(True):
            cond = (arr[i] == gen).all()
        if cond:
            return i
    return -1


def next_generation(gen):
    next_gen = {}
    selec = selection(gen)
    parents = pairing(selec)
    offsprings = [[[mating(parents[x])
                    for x in range(len(parents))]
                   [y][z] for z in range(2)]
                  for y in range(len(parents))]
    offspr1 = [offsprings[x][0]
               for x in range(len(parents))]
    offspr2 = [offsprings[x][1]
               for x in range(len(parents))]
    offspr1.extend(offspr2)
    unmutated = offspr1

    # Решили сносить родителей до мутаций
    for i in range(len(parents)):
        ind = eqGen(gen['Individuals'], parents[i][0])
        if ind != -1 and fitness_calculation(offspr1[i]) > 0:
            gen['Individuals'][ind] = offspr1[i]
        ind = eqGen(gen['Individuals'], parents[i][1])
        if ind != -1 and fitness_calculation(offspr2[i]) > 0:
            gen['Individuals'][ind] = offspr2[i]
    mutated = mutation(unmutated)
    unsorted_individuals = mutated

    mutated.extend(gen['Individuals'])

    unsorted_next_gen = [fitness_calculation(mutated[x]) for x in range(len(mutated))]
    sorted_next_gen = sorted(
        [[unsorted_individuals[x], unsorted_next_gen[x]] for x in range(len(unsorted_individuals))],
        key=lambda x: x[1])
    next_gen['Individuals'] = [sorted_next_gen[x][0] for x in range(len(sorted_next_gen))]
    next_gen['Fitness'] = [sorted_next_gen[x][1] for x in range(len(sorted_next_gen))]
    return next_gen


def first_generation(pop):
    fitness = [fitness_calculation(pop[x])
               for x in range(len(pop))]
    sorted_fitness = sorted([[pop[x], fitness[x]]
                             for x in range(len(pop))], key=lambda x: x[1])
    population = [sorted_fitness[x][0]
                  for x in range(len(sorted_fitness))]
    fitness = [sorted_fitness[x][1]
               for x in range(len(sorted_fitness))]
    return {'Individuals': population, 'Fitness': sorted(fitness)}


# сгенирировать популяцию 200 - кол во элементов 30- кол во генов
pop = population(200, 30)

gen = [first_generation(pop)]
fitness_max = np.array([max(gen[0]['Fitness'])])

finish = 0


# сходимость или прошло 500 поколений другое решение,
# при первой итерации находится особь с хорошим приспособлением - алгоритм оканчивается, когда разница между функциями приспособленности больше, чем стоимость самой дешевой вещи
def fitness_max_chech(max_fitness):
    result = 0
    if len(max_fitness) > 2:
        for n in range(len(max_fitness) - 1):
            result = max_fitness[n] - max_fitness[n-1]
    return result



while finish < 500:
    diff_max = fitness_max_chech(fitness_max)
    if finish > 1 and diff_max <= min(capacity):
        print("Разница:", diff_max, "Самая дешевая вещь:", min(capacity))
        break

    gen.append(next_generation(gen[-1]))
    finish = finish + 1
    fitness_max = np.append(fitness_max, max(gen[-1]['Fitness']))

solution = gen[-1]['Individuals'][num_items - 1]
print(solution)
solution_fitness = gen[-1]['Fitness'][num_items - 1]
print('№   Вес   Объем  Ценность')
total_weight = 0.0
total_value = 0.0
for i in range(len(solution)):
    total_value += solution[i] * value[i]
    total_weight += solution[i] * weight[i]
    if solution[i] == 1:
        print('{0} {1} {2} {3}\n'.format(i, weight[i], value[i], capacity[i], ))
print("Вес:", total_weight, "Объем:", total_value, "Ценность:", solution_fitness)
