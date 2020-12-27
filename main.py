import multiprocessing
import urllib.request
import time
import cplex
import random
import math
import numpy as np
import copy

# GLOBAL
timeLimitValue = 3600
delta = 0.00001
local = True
bestDecision = 0
maxColorGlobal = []
grath = {}
coloredEdValue = []

localPaths = [
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphsBNP/myciel3.col',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphsBNP/queen5_5.col'
]


# --------------------OPEN FILE--------------------
def openGraph(filePath):
    n = -1
    m = -1
    global local
    if local == True:
        file = open(filePath)
    else:
        file = urllib.request.urlopen(filePath)
    for line in file:
        if local == False:
            line = line.decode('ascii')
            line = line.strip('\n')
        if line.startswith('p'):
            n = int(line.split(' ')[2])
            m = int(line.split(' ')[3])
            break
    graphMatrix = np.zeros((n, n))
    for line in file:
        if local == False:
            line = line.decode('ascii')
            line = line.strip('\n')
        if line.startswith('e'):
            i = int(line.split(' ')[1]) - 1
            j = int(line.split(' ')[2]) - 1
            graphMatrix[i, j] = 1
            graphMatrix[j, i] = 1
    return n, m, graphMatrix


def graphByNeighborhoods(graphMatrix, n):
    graphModel = [[] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and graphMatrix[i][j] == 1 and j not in graphModel[i]:
                graphModel[i].append(j)
    return graphModel


# ---------------------------------------------------------------------
# ----------------------Heuristic Functions--------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# красим жадно
def colorGreedy(matrix, vertex):
    V = [i for i in range(vertex)]
    colorGroups = [[]]
    coloredV = [-1 for p in range(vertex)]
    k = 0
    for i in range(vertex):
        if i not in V:
            continue
        colorGroups[k].append(i)
        V.remove(i)
        while len(matrix[i].nonzero()[0]) != vertex:  # пока есть ненулевые
            for j in range(i, vertex):  # ?
                if matrix[i, j] == 0 and j in V:
                    break
            if j == vertex:
                break
            if j == vertex - 1 and matrix[i, j] != 0 or j not in V:
                break

            colorGroups[k].append(j)
            V.remove(j)
            matrix[i] = matrix[i] + matrix[j]

        k = k + 1
        colorGroups.insert(k, [])
    if len(colorGroups[len(colorGroups) - 1]) == 0:
        colorGroups.pop()
    for i in range(k):
        for j in range(len(colorGroups[i])):
            coloredV[colorGroups[i][j]] = i
    return coloredV, colorGroups


def evristicBNP(matrix, n, path):
    print('heuristics start for: ', path)

    coloredEd, colorGroups = colorGreedy(matrix.copy(), n)

    colorsHolder = [[] for i in range(n)]

    for i in range(1):
        colorGroupsTmp = copy.deepcopy(colorGroups)
        for index, group in enumerate(colorGroupsTmp):
            for i in group:
                colorsHolder[i].append(index)
        for index, group in enumerate(colorGroupsTmp):
                for j in range(n):
                    flag = True
                    for i in group:
                        if j in group or matrix[i][j] == 1:
                            flag = False
                    if flag == True and index not in colorsHolder[j]:
                        for z in group:
                            tmp = list(set(colorsHolder[z]) & set(colorsHolder[j]))
                            if len(tmp) > 0:
                                flag = False
                    if flag == True:
                        colorsHolder[j].append(index)
                        group.append(j)
        for index, vert in enumerate(colorsHolder):
            print(index, ' ', vert)
        colorGroups = colorGroupsTmp
    return colorsHolder, colorGroups


# ---------------------------------------------------------------------
def indSetSearch(neighborsGraph, weight):
    maxInitialValue = 0
    maxInitialValueVertex = 0

    for i in range(len(weight)):
        if weight[i] >= maxInitialValue:
            maxInitialValue = weight[i]
            maxInitialValueVertex = i
    sumMax = 0
    indSet = [maxInitialValueVertex]
    indSetMax = []


    for i in range(50):
        sum, indSetNew = localSearch(neighborsGraph, copy.copy(indSet), copy.copy(weight))
        if sumMax < sum:
            sumMax = sum
            indSetMax = copy.copy(indSetNew)

    for i in indSetMax:
        for j in indSetMax:
            if i != j and i in neighborsGraph[j]:
                print('ERROR IND SET 2', indSet, indSetMax)
                break
    return sumMax, indSetMax

# локальный поиск
def localSearch(graphNeighbors, indSet, weight):
    indSetOld = copy.copy(indSet)

    N = len(graphNeighbors)
    # statusArray:
    # 1 - indSet
    # 2 - freeVertex
    # 3 - bindedVertex
    tightness = [0 for i in range(N)]
    statusArray = [2 for i in range(N)]
    for i in indSet:
            statusArray[i] = 1
            for j in graphNeighbors[i]:
                tightness[j] += 1
                statusArray[j] = 3
    candidatsVertex = []
    for item in indSet:
        tightCount = 0
        for j in graphNeighbors[item]:
            if tightness[j] == 1:
                tightCount += 1
                if tightCount >= 2:
                    candidatsVertex.append(item)
                    break
    freeVertex = []
    for i in range(len(statusArray)):
        if statusArray[i] == 2:
            freeVertex.append(i)

    while len(candidatsVertex) > 0 or len(freeVertex)>0:
        if len(candidatsVertex) == 0:
            newVertex = random.choice(freeVertex)
            candidatsVertex.append(newVertex)
            statusArray[newVertex] = 1
            freeVertex.remove(newVertex)
            for val in graphNeighbors[newVertex]:
                tightness[val] += 1
        vertForSwap = random.choice(candidatsVertex)
        u = -1
        v = -1
        for i in graphNeighbors[vertForSwap]:
            if tightness[i] == 1:
                if u == -1:
                    u = i
                else:
                    if i not in graphNeighbors[u]:
                        v = i
            if u != -1 and v != -1:
                break
        if u != -1 and v != -1 and (weight[u]+weight[v]) >= weight[vertForSwap]:
            statusArray[u] = statusArray[v] = 1
            statusArray[vertForSwap] = 3

            candidatsVertex.remove(vertForSwap)
            for j in graphNeighbors[vertForSwap]:
                tightness[j] = tightness[j]- 1

            for j in graphNeighbors[u]:
                tightness[j] += 1


            for j in graphNeighbors[v]:
                tightness[j] += 1

            for j in range(len(statusArray)):
                if tightness[j] <= 0 and statusArray[j] != 1:
                    statusArray[j] = 2
                if tightness[j] >= 1:
                    statusArray[j] = 3
            candidatsVertex = []
            for j in range(len(statusArray)):
                if statusArray[j]==1:
                    tightCount = 0
                    for j1 in graphNeighbors[j]:
                        if tightness[j1] == 1:
                            tightCount += 1
                            if tightCount >= 2:
                                candidatsVertex.append(j)
                                break
            freeVertex = []
            for j in range(len(statusArray)):
                if statusArray[j] == 2 and tightness[j] == 0:
                    freeVertex.append(j)
        else:
            freeVertex = []
            for j in range(len(statusArray)):
                if statusArray[j] == 2 and tightness[j] == 0:
                    freeVertex.append(j)
            candidatsVertex.remove(vertForSwap)
            continue
    sumWeight = 0
    answer = []
    sumWeightOld = 0
    for i in indSetOld:
        sumWeightOld = sumWeightOld + weight[i]
    for i in range(len(statusArray)):
        if statusArray[i] == 1:
            sumWeight += weight[i]
            answer.append(i)
    if (sumWeightOld> sumWeight):
        return sumWeightOld, indSetOld
    return sumWeight, answer

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# Инициализируем модель cplex (добавляем все ограничения)
def initalMasterCPLEX(masterModel, colorsHolder, colorGroups):
    colorsCount = len(colorGroups)
    masterModel.variables.add(names=["y" + str(i) for i in range(colorsCount)])

    for i in range(colorsCount):
        masterModel.variables.set_lower_bounds(i, 0.0)

    masterModel.set_log_stream(None)
    masterModel.set_warning_stream(None)
    # masterModel.set_error_stream(None)
    masterModel.set_results_stream(None)

    for i in range(colorsCount):
        masterModel.objective.set_linear("y" + str(i), 1)

    masterModel.objective.set_sense(masterModel.objective.sense.minimize)

    for vertColors in colorsHolder:
        constrainsName = "constraint"
        constrains = []
        constrainsNames = []
        constrainsTypes = []
        constrainsRightParts = []
        variables = []
        for color in vertColors:
            variables.append("y" + str(color))
            constrainsName = constrainsName + "_" + str(color)
        coef = [1] * len(vertColors)
        constrains.append([variables])
        constrainsNames.append(constrainsName)
        constrainsTypes.append('G')
        constrainsRightParts.append(1.0)
        masterModel.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=variables, val=coef)],
            rhs=constrainsRightParts,
            names=constrainsNames,
            senses=constrainsTypes)

    masterModel.solve()
    values = masterModel.solution.get_values()
    return values, masterModel.solution.get_objective_value()


# ------------------------BNP----------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# функция, которую запускаем в отедльном процессе, чтобы была возможность остановить по времени
def bnpContainer(colorsHolder, colorGroups, n, return_dict, coloredEd, grathNeighborsV, confusion_matrix):
    global bestDecision
    global coloredEdValue
    global grath

    coloredEdValue = coloredEd
    grath = grathNeighborsV

    masterModel = cplex.Cplex()

    bestDecision, bestResult = initalMasterCPLEX(masterModel, colorsHolder, colorGroups)

    print('bestDecision ', bestDecision)
    global timer
    timer = 0
    result, resultValues = BNP(colorGroups, colorsHolder, masterModel)
    print('')
    print('')
    print('')
    print('')

#     проход по всем графам из файлов и запуск эвристики и bnc для каждого
def bnpStartEngine(graphs):
    for i in range(len(graphs)):
        global grath
        global coloredEdValue
        n, m, confusion_matrix = openGraph(graphs[i])
        grath = graphByNeighborhoods(confusion_matrix, len(confusion_matrix))
        colorsHolder, colorGroups = evristicBNP(confusion_matrix, n, graphs[i])

        # check decision
        #
        if __name__ == '__main__':
            global timeLimitValue
            manager = multiprocessing.Manager()
            print('start bnP for ', graphs[i])
            p = multiprocessing.Process(target=bnpContainer,
                                        args=(colorsHolder, colorGroups, n, {}, coloredEdValue, grath, confusion_matrix))
            p.start()
            p.join(timeLimitValue)
            if p.is_alive():
                p.terminate()

def columnGeneration(masterModel, grath, colorGroups, colorsHolder):
    dualValues = masterModel.solution.get_dual_values(0, len(grath)-1)
    for i in range(len(dualValues)):
        if dualValues[i] <=0:
            dualValues[i]=0.00000001
    sumMax, indSetMax = indSetSearch(copy.copy(grath), dualValues)
    return sumMax, indSetMax

def columnGenerationLoop(masterModel, C, exact, grath, colorGroups, colorsHolder, currentDecisionValue):
    sumMax, indSetMax = columnGeneration(masterModel, grath, colorGroups, colorsHolder)
    badLoop = 0
    while sumMax > 1 and badLoop < 10:
        colorGroups.append(indSetMax)
        for i in indSetMax:
            colorsHolder[i].append(len(colorGroups)-1)
        masterModel = cplex.Cplex()
        bestDecision, bestResult = initalMasterCPLEX(masterModel, colorsHolder, colorGroups)

        print('sumMax indSetMax ', sumMax, ' ', indSetMax)
        if abs(currentDecisionValue - bestResult) < 0.01:
            badLoop = badLoop + 1
        currentDecisionValue = bestResult
        sumMax, indSetMax = columnGeneration(masterModel, grath, colorGroups, colorsHolder)
    return True

# BNP
def BNP(colorGroups, colorsHolder, masterModel):

    currentResult = masterModel.solution.get_objective_value()
    C = []
    not_tailing_off = columnGenerationLoop(masterModel, C, False, grath, colorGroups, colorsHolder, currentResult)
    # to be continued ...
    return [], []

# MAIN
if __name__ == '__main__':
    bnpStartEngine(localPaths)

