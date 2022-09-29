import numpy as np

if __name__ == "__main__":
    # import facebook data
    def import_facebook_data(path):
        f = open(path, 'r')
        lines = f.readlines()
        lines = [ [ int(j) for j in i.split() ] for i in lines ]
        maxNode = -1
        minNode = len(lines)
        for i in range(len(lines)):
            for j in range(2):
                if maxNode < lines[i][j]:
                    maxNode = lines[i][j]
                if minNode > lines[i][j]:
                    minNode = lines[i][j]
        adjMat = np.zeros((maxNode + 1, maxNode + 1))
        def edgeCheck(l):
            if adjMat[l[0]][l[1]] == 1:
                return False
            else:
                adjMat[l[0]][l[1]] = 1
                adjMat[l[1]][l[0]] = 1
                return True
        uniEdges = [ i for i in lines if edgeCheck(i) ]
        revEdges = [ [i[1], i[0]] for i in uniEdges ]
        return np.array(uniEdges + revEdges)
    
    # spectral decomp oneIter
    def spectralDecomp_OneIter(nodes_connectivity_list):
        def getMinNodeId(nodeRevMap, nodesList):
            minId = float('inf')
            for i in nodesList:
                if nodeRevMap[i] < minId:
                    minId = nodeRevMap[i]
            return minId
        def getNodes(nodes_connectivity_list):
            nodes = [ ]
            for i in range(len(nodes_connectivity_list)):
                if nodes_connectivity_list[i][0] not in nodes:
                    nodes.append(nodes_connectivity_list[i][0])
                if nodes_connectivity_list[i][1] not in nodes:
                    nodes.append(nodes_connectivity_list[i][1])
            # nodes = sorted(nodes)
            return ({ n: i for n, i in zip(nodes, range(len(nodes))) }, { i: n for n, i in zip(nodes, range(len(nodes))) })
        def getAdjMat(nodes_connectivity_list, nodeMap, n):
            adjMat = np.zeros((n, n))
            for l in nodes_connectivity_list:
                adjMat[nodeMap[l[0]]][nodeMap[l[1]]] = 1
                adjMat[nodeMap[l[1]]][nodeMap[l[0]]] = 1
            return adjMat
        def disconnectedComponents(adjMat, n):
            marks = np.zeros((n))
            marksDone = 0
            components = [ ]
            while len(marks) > marksDone:
                currNode = np.argmin(marks)
                nodeStack = [ currNode ]
                marks[currNode] = 1
                marksDone = marksDone + 1
                components.append([ ])
                while len(nodeStack) > 0:
                    currNode = nodeStack.pop()
                    components[len(components) - 1].append(currNode)
                    for i in range(n):
                        if adjMat[currNode][i] == 1 and marks[i] == 0:
                            nodeStack.append(i)
                            marks[i] = 1
                            marksDone = marksDone + 1
            return components
        def getNormalizedLaplacian(adjMat, n):
            degreeDiag = adjMat.sum(axis=1).flatten()
            degreeMat = np.zeros((n, n))
            for i, j in zip(degreeDiag, range(n)):
                degreeMat[j][j] = i
            L = degreeMat - adjMat
            with np.errstate(divide="ignore"):
                degreeDiagSqrt = 1.0 / np.sqrt(degreeDiag)
            degreeDiagSqrt[np.isinf(degreeDiagSqrt)] = 0
            degreeSqrtMat = np.zeros((n, n))
            for i, j in zip(degreeDiagSqrt, range(n)):
                degreeSqrtMat[j][j] = i
            return degreeSqrtMat @ (L @ degreeSqrtMat)
        def getFiedler(adjMat, n):
            normalizedLaplacian = getNormalizedLaplacian(adjMat, n)
            eigVal, eigVec = np.linalg.eigh(normalizedLaplacian)
            eigenVectorSorted = [ ( i[0], i[1] ) for i in zip(eigVec.real.T, eigVal.real) ]
            eigenVectorSorted = sorted(eigenVectorSorted, key = lambda x: x[1])
            eigVal = eigenVectorSorted[1][1]
            eigenVectorSorted = np.array(list(map(lambda x: x[0], eigenVectorSorted)))
            return (eigenVectorSorted[1], eigVal)
        # map n -> i, rev map i -> n
        nodeMap, nodeRevMap = getNodes(nodes_connectivity_list)
        n = len(nodeMap)
        fiedlerVectorFinal = np.zeros((n))
        adjMat = getAdjMat(nodes_connectivity_list, nodeMap, n)
        disComp = disconnectedComponents(adjMat, n)
        currColor = 0
        retPartitionSet = [ ]
        # print('number of partition', len(disComp), [ len(i) for i in disComp ])
        for disc in disComp:
            # print('started', currColor)
            def getAdjMat2(adjMat, compNodes):
                m = len(compNodes)
                newAdjMat = np.zeros((m , m))
                for i, ii in zip(compNodes, range(m)):
                    for j, ji in zip(compNodes, range(m)):
                        newAdjMat[ii][ji] = adjMat[i][j]
                return newAdjMat
            def getGraphPartitionSet(disc, initVal):
                return np.array([ [ i, initVal ] for i in disc ])
            graphPartitionSet = getGraphPartitionSet(disc, 0)
            adjMatComp = getAdjMat2(adjMat, disc)
            m = len(disc)
            if m < 2:
                # print('stopped1')
                tColor = getMinNodeId(nodeRevMap, disc)
                for i in disc:
                    nodeMap[nodeRevMap[i]] = tColor
                continue
            fiedlerVector, _ = getFiedler(adjMatComp, m)
            for f, i in zip(fiedlerVector, disc):
                fiedlerVectorFinal[i] = f
            posVals = 0
            posCount = 0
            negVals = 0
            negCount = 0
            posNodesMin = float('inf')
            negNodesMin = float('inf')
            for i in range(m):
                if fiedlerVector[i] <= 0:
                    negVals = negVals + (fiedlerVector[i] * -1)
                    negCount = negCount + 1
                    if negNodesMin > nodeRevMap[disc[i]]:
                        negNodesMin = nodeRevMap[disc[i]]
                else:
                    graphPartitionSet[i][1] = 1
                    posVals = posVals + (fiedlerVector[i])
                    posCount = posCount + 1
                    if posNodesMin > nodeRevMap[disc[i]]:
                        posNodesMin = nodeRevMap[disc[i]]
            ratio = abs(posVals / posCount - negVals / negCount)
            if posCount == 0 or negCount == 0:
                # print('stopped2')
                tColor = getMinNodeId(nodeRevMap, disc)
                for i in disc:
                    nodeMap[nodeRevMap[i]] = tColor
                currColor = currColor + 1
            elif ratio > 0.05:
                # print('stopped3')
                tColor = getMinNodeId(nodeRevMap, disc)
                for i in disc:
                    nodeMap[nodeRevMap[i]] = tColor
                currColor = currColor + 1
            else:
                # print('did partition', ratio)
                for i in graphPartitionSet:
                    nodeMap[nodeRevMap[i[0]]] = posNodesMin if i[1] == 1 else negNodesMin
                currColor = currColor + 2
        graphPartition = np.array([ [ i, nodeMap[i] ] for i in nodeMap ])
        return (fiedlerVectorFinal * -1, adjMat, graphPartition)
    
    # spectral decomposition
    def spectralDecomposition(nodes_connectivity_list):
        _, adj_mat, graph_partition = spectralDecomp_OneIter(nodes_connectivity_list)
        def getNodeClusters(graph_partition):
            m_map = { }
            m = [ ]
            for i in graph_partition:
                if i[1] not in m:
                    m_map[i[1]] = len(m)
                    m.append(i[1])
            m = [ [ ] for i in range(len(m)) ]
            for i in graph_partition:
                m[m_map[i[1]]].append(i[0])
            return m
        nodeClusters = getNodeClusters(graph_partition)
        if len(nodeClusters) == 1:
            return graph_partition
        def getNodes(nodes_connectivity_list):
            nodes = [ ]
            for i in range(len(nodes_connectivity_list)):
                if nodes_connectivity_list[i][0] not in nodes:
                    nodes.append(nodes_connectivity_list[i][0])
                if nodes_connectivity_list[i][1] not in nodes:
                    nodes.append(nodes_connectivity_list[i][1])
            # nodes = sorted(nodes)
            return ({ n : i for n, i in zip(nodes, range(len(nodes))) }, { i: n for n, i in zip(nodes, range(len(nodes))) })
        # n -> i, i -> n
        nodeMap, nodeRevMap = getNodes(nodes_connectivity_list)
        nodesRevMap2 = { int(i[0]): j for i, j in zip(graph_partition, range(len(graph_partition))) }
        nodes_connectivity_list_new_s = [ ]
        for cl in nodeClusters:
            nodes_connectivity_list_new_s.append([ ])
            for i in cl:
                for j in cl:
                    if adj_mat[nodeMap[i]][nodeMap[j]] == 1:
                        nodes_connectivity_list_new_s[len(nodes_connectivity_list_new_s) - 1].append([i, j])
        
        for cli in range(len(nodeClusters)):
            spectralPart = spectralDecomposition(nodes_connectivity_list_new_s[cli])
            for i in spectralPart:
                graph_partition[nodesRevMap2[i[0]]][1] = i[1]
        return graph_partition
    
    # sorted adj matrix
    def createSortedAdjMat(graph_partition, nodes_connectivity_list):
        graph_partition = graph_partition.copy()
        graph_partition = sorted(graph_partition, key = lambda x: x[1])
        def getNodes(nodes_connectivity_list):
            nodes = [ ]
            for i in range(len(nodes_connectivity_list)):
                if nodes_connectivity_list[i][0] not in nodes:
                    nodes.append(nodes_connectivity_list[i][0])
                if nodes_connectivity_list[i][1] not in nodes:
                    nodes.append(nodes_connectivity_list[i][1])
            nodes = sorted(nodes)
            return ({ i: j for i, j in zip(nodes, range(len(nodes))) }, { j: i for i, j in zip(nodes, range(len(nodes))) })
        def getAdjMat(nodes_connectivity_list, nodeMap, n):
            adjMat = np.zeros((n, n))
            for l in nodes_connectivity_list:
                adjMat[nodeMap[l[0]]][nodeMap[l[1]]] = 1
            return adjMat
        nodeMap, nodeRevMap = getNodes(nodes_connectivity_list)
        n = len(nodeMap)
        adjMat = getAdjMat(nodes_connectivity_list, nodeMap, n)
        
        sortedAdjMat = np.zeros((n , n))
        for i, ii in zip(graph_partition, range(n)):
            for j, ji in zip(graph_partition, range(n)):
                sortedAdjMat[ii][ji] = adjMat[nodeMap[i[0]]][nodeMap[j[0]]]
        return sortedAdjMat
    
    # louvain algorithm
    def louvain_one_iter(nodes_connectivity_list):
        def getMinNodeId(nodeRevMap, nodesList):
            minId = float('inf')
            for i in nodesList:
                if nodeRevMap[i] < minId:
                    minId = nodeRevMap[i]
            return minId
        def getNodes(nodes_connectivity_list):
            nodes = [ ]
            for i in range(len(nodes_connectivity_list)):
                if nodes_connectivity_list[i][0] not in nodes:
                    nodes.append(nodes_connectivity_list[i][0])
                if nodes_connectivity_list[i][1] not in nodes:
                    nodes.append(nodes_connectivity_list[i][1])
            # nodes = sorted(nodes)
            return ({ n: i for n, i in zip(nodes, range(len(nodes))) }, { i: n for n, i in zip(nodes, range(len(nodes))) })
        def getAdjMat(nodes_connectivity_list, nodeMap, n):
            adjMat = np.zeros((n, n))
            for l in nodes_connectivity_list:
                adjMat[nodeMap[l[0]]][nodeMap[l[1]]] = 1
                adjMat[nodeMap[l[1]]][nodeMap[l[0]]] = 1
            return adjMat
        def getQ(A, D, m):
            n = len(A)
            Q = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    Q[i][j] = A[i][j] - D[i] * D[j] / (2 * m)
            Q = Q / (2 * m)
            return Q
        def getDelQ(Q, i, X):
            Qbefore = Q[i][i] + X[1]
            addedDQ = Q[i][i]
            if i not in X[0]:
                for j in X[0]:
                    addedDQ = addedDQ + Q[j][i] + Q[i][j]
            Qafter = X[1] + addedDQ
            return (Qafter - Qbefore, addedDQ)
        def louvainItter(Q, C, Cmap):
            n = len(Q)
            totalUpdates = 0
            for i in range(n):
                i_n = i
                j_n = -1
                maxDQ = float('-inf')
                addedDQ_n = 0
                for j in C:
                    cDQ, cAddedDQ = getDelQ(Q, i, C[j])
                    if cDQ > maxDQ:
                        maxDQ = cDQ
                        addedDQ_n = cAddedDQ
                        j_n = j
                        i_n = i
                if j_n != Cmap[i]:
                    totalUpdates = totalUpdates + maxDQ
                    removeDQ = 0
                    for j in C[Cmap[i_n]][0]:
                        removeDQ = removeDQ + Q[j][i_n] + Q[i_n][j]
                    C[Cmap[i_n]][1] = C[Cmap[i_n]][1] - removeDQ
                    C[Cmap[i_n]][0].remove(i_n)
                    if len(C[Cmap[i_n]][0]) == 0:
                        C.pop(Cmap[i_n])
                    Cmap[i_n] = j_n
                    C[j_n][0].add(i_n)
                    C[j_n][1] = C[j_n][1] + addedDQ_n
            return (round(totalUpdates, 5), len(C))
        # n -> i, i -> n
        nodeMap, nodeRevMap = getNodes(nodes_connectivity_list)
        n = len(nodeMap)
        adjMat = getAdjMat(nodes_connectivity_list, nodeMap, n)
        degreeList = [ sum(i) for i in adjMat ]
        edgesCount = int(sum(degreeList) / 2)
        Q = getQ(adjMat, degreeList, edgesCount)
        C = { i: [{i}, Q[i][i]] for i in range(len(Q)) }
        Cmap = [ i for i in range(len(Q)) ]
        nodeIndex = [ [ i ] for i in range(len(Q)) ]
        tlList, tcMin = ([ ], len(Q) + 1)
        tln, tcn = louvainItter(Q, C, Cmap)
        while tln not in tlList:
            # print(tlList, tln, tcn)
            tlList.append(tln)
            if tcMin != tcn:
                tlList = [ ]
            tcMin = min(tcMin, tcn)
            tln, tcn = louvainItter(Q, C, Cmap)
        graph_partition = [ ]
        for c in C:
            minNode = float('inf')
            for i in C[c][0]:
                if minNode > nodeRevMap[i]:
                    minNode = nodeRevMap[i]
            for i in C[c][0]:
                graph_partition.append([ nodeRevMap[i], minNode ])
        return np.array(graph_partition)
    
    # read bitcoin data
    def import_bitcoin_data(path):
        f = open(path, 'r')
        lines = f.readlines()
        lines = [ [ int(j) for j in i[ : -1].split(',')[ : 2] ] for i in lines ]
        maxNode = -1
        minNode = len(lines)
        for i in range(len(lines)):
            for j in range(2):
                if maxNode < lines[i][j]:
                    maxNode = lines[i][j]
                if minNode > lines[i][j]:
                    minNode = lines[i][j]
        adjMat = np.zeros((maxNode + 1, maxNode + 1))
        def edgeCheck(l):
            if adjMat[l[0]][l[1]] == 1:
                return False
            else:
                adjMat[l[0]][l[1]] = 1
                adjMat[l[1]][l[0]] = 1
                return True
        uniEdges = [ i for i in lines if edgeCheck(i) ]
        revEdges = [ [i[1], i[0]] for i in uniEdges ]
        return np.array(uniEdges + revEdges)
    
    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")

    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)

    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communitites.
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)


    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import soc-sign-bitcoinotc.csv
    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")

    # Question 1
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)

    # Question 2
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)

    # Question 3
    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    # Question 4
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)