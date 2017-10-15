from __future__ import division
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pickle

def infect(network, p, seednode, sorted_data):
    infected = dict(zip(nodes, np.zeros(len(nodes))))
    infectionTimes = dict(zip(nodes, np.zeros(len(nodes)) + np.inf))
    infected[np.unicode(seednode)] = 1
    infectionTimes[np.unicode(seednode)] = sorted_data[0][2]
    for i in sorted_data:
        curTime = i[2]
        if infected[np.unicode(i[0])] == 1 and curTime >= infectionTimes[np.unicode(i[0])]:
            if np.random.rand() < p: 
                infected[np.unicode(i[1])] = 1
                if i[3] < infectionTimes[np.unicode(i[1])]:
                    infectionTimes[np.unicode(i[1])] = i[3]
    return infected, infectionTimes

def getData(network):
    nodes = network.nodes()
    degree = np.array([nx.degree(network)[x] for x in nodes])
    bc = np.array([nx.betweenness_centrality(network)[x] for x in nodes])
    closeness = np.array([nx.closeness_centrality(network)[x] for x in nodes])
    strength = np.array([nx.degree(network, nodes, weight = "weight")[x] for x in nodes])
    cc = np.array([nx.clustering(network, nodes)[x] for x in nodes])
    kshell = nx.degree(network)
    for k in range(1, max(degree) + 1):
        curShell = nx.k_shell(network, k) 
        for i in range(0, len(curShell)):
            kshell[curShell.nodes()[i]] = k 
    kshell = np.array([kshell[x] for x in nodes])  
    return kshell, cc, degree, strength, bc, closeness

def ImmunizeInf(network, p, seednode, sorted_data, immunizeInds):
    infected = dict(zip(nodes, np.zeros(len(nodes))))
    infectionTimes = dict(zip(nodes, np.zeros(len(nodes)) + np.inf))
    infected[np.unicode(seednode)] = 1
    infectionTimes[np.unicode(seednode)] = sorted_data[0][2]
    immunizeInds = np.array(immunizeInds)
    for i in sorted_data:
        curTime = i[2]
        if sum(immunizeInds == i[1]) == 0 and sum(immunizeInds == i[0]) == 0 and infected[np.unicode(i[0])] == 1 and curTime >= infectionTimes[np.unicode(i[0])]:
            if np.random.rand() < p: 
                infected[np.unicode(i[1])] = 1
                if i[3] < infectionTimes[np.unicode(i[1])]:
                    infectionTimes[np.unicode(i[1])] = i[3]
    return infected, infectionTimes

def infectLinks(network, p, seednode, sorted_data):
    infected = dict(zip(nodes, np.zeros(len(nodes))))
    infectionLinks = dict([(key, []) for key in  nodes])
    infectionTimes = dict(zip(nodes, np.zeros(len(nodes)) + np.inf))
    infected[np.unicode(seednode)] = 1
    infectionTimes[np.unicode(seednode)] = sorted_data[0][2]
    for i in sorted_data:
        curTime = i[2]
        if infected[np.unicode(i[0])] == 1 and curTime >= infectionTimes[np.unicode(i[0])]:
            if np.random.rand() < p: 
                infected[np.unicode(i[1])] = 1
                if i[3] < infectionTimes[np.unicode(i[1])]:
                    infectionTimes[np.unicode(i[1])] = i[3]
                    infectionLinks[np.unicode(i[1])] = (np.unicode(i[0]))
    return infected, infectionTimes, infectionLinks

def plot_network_usa(net, xycoords, weights=1, edges=None, alpha=0.2):

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 0.9])
    # ([0, 0, 1, 1])
    bg_figname = 'C:/Users/danie/Desktop/ProjectWork/US_air_bg.png'
    img = plt.imread(bg_figname)
    axis_extent = (-6674391.856090588, 4922626.076444283,
                   -2028869.260519173, 4658558.416671531)
    ax.imshow(img, extent=axis_extent)
    ax.set_xlim((axis_extent[0], axis_extent[1]))
    ax.set_ylim((axis_extent[2], axis_extent[3]))
    ax.set_axis_off()
    nx.draw_networkx(net,
                     pos=xycoords,
                     with_labels=False,
                     node_color='k',
                     node_size=5,
                     edge_color='r',
                     alpha=alpha,
                     edgelist=edges,
                     width = weights)
    return fig, ax

def getEdges(network, edges):
    overlap = []
    weights = []
    link_bc = []
    bc = nx.edge_betweenness_centrality(network)
    for edge in network.edges():
        ki = nx.degree(network, edge[0])
        kj = nx.degree(network, edge[1])
        nij = len(sorted(nx.common_neighbors(network, edge[0], edge[1])))
        denom = (ki - 1) + (kj - 1) - nij
        if denom == 0:
            overlap.append(0)
        else:
            overlap.append(nij / denom)
        weights.append(network[edge[0]][edge[1]]["weight"])
        link_bc.append(bc[edge])
    return [overlap, weights, link_bc]

if __name__ == "__main__":
    event_data = np.genfromtxt('C:/Users/danie/Desktop/ProjectWork/events_US_air_traffic_GMT.txt', names = True, dtype = int, delimiter = ' ')
    network = nx.read_weighted_edgelist('C:/Users/danie/Desktop/ProjectWork/aggregated_US_air_traffic_network_undir.edg')
    sorted_data = np.sort(event_data, order='StartTime')
    nodes = network.nodes()

    #######
    # Task 1 
    ####### 
    
    infected, infectionTimes = infect(nodes, 1, 0, sorted_data)
    print("Anchorage infected: " + str(infectionTimes[u'41']))

    #######
    # Task 2
    ####### 
     
    n = 10
    ps = np.array([0.01, 0.05, 0.1, 0.5, 1])
    infectionTimesList = []
    """
    for p in ps:
        for i in range(0, n):
            _, infectionTimes = infect(nodes, p, 0, sorted_data)
            infectionTimesList.append(infectionTimes)

    pickle.dump(infectionTimesList, open("C:/Users/danie/Desktop/ProjectWork/templist.txt","wb"))
    """
    infectionTimesList = pickle.load(open("C:/Users/danie/Desktop/ProjectWork/templist.txt","rb"))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    timeStart = sorted_data[0][2]
    timeStop = np.sort(sorted_data, order='EndTime')[-1][3]
    timeRange = range(timeStart, timeStop, 1000)
    for i in range(0, len(ps)):
        fracInfected = []
        curInfectionTimes = []
        for k in range(i*10, (i+1)*10):
            curInfectionTimes.append(infectionTimesList[k].values())
        curInfectionTimes = [item for sublist in curInfectionTimes for item in sublist]
        curInfectionTimes = np.array(curInfectionTimes)
        numEl = len(curInfectionTimes)
        for j in timeRange:
            fracInfected.append(sum(curInfectionTimes<j)/numEl)
        ax.plot(timeRange, fracInfected, label="p = " + str(ps[i]))
    ax.legend(loc=4)
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction of infected airports')
            
    #######
    # Task 3 
    ####### 
    
    n = 10
    p = 0.1
    nodeIDs = np.array([0, 4, 41, 100, 200])
    """
    infectionTimesList = []
    for nodeID in nodeIDs:
        for i in range(0, n):
            _, infectionTimes = infect(nodes, p, nodeID, sorted_data)
            infectionTimesList.append(infectionTimes)

    pickle.dump(infectionTimesList, open("C:/Users/danie/Desktop/ProjectWork/templist2.txt","wb"))
    """
    infectionTimesList = pickle.load(open("C:/Users/danie/Desktop/ProjectWork/templist2.txt","rb"))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    timeStart = sorted_data[0][2]
    timeStop = np.sort(sorted_data, order='EndTime')[-1][3]
    timeRange = range(timeStart, timeStop, 1000)
    for i in range(0, len(nodeIDs)):
        fracInfected = []
        curInfectionTimes = []
        for k in range(i*10, (i+1)*10):
            curInfectionTimes.append(infectionTimesList[k].values())
        curInfectionTimes = [item for sublist in curInfectionTimes for item in sublist]
        curInfectionTimes = np.array(curInfectionTimes)
        numEl = len(curInfectionTimes)
        for j in timeRange:
            fracInfected.append(sum(curInfectionTimes<j)/numEl)
        ax.plot(timeRange, fracInfected, label="node = " + str(nodeIDs[i]))
    ax.legend(loc=4)
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction of infected airports')
    
    #######
    # Task 4 
    ####### 
    
    numNodes = len(nodes)
    p = 0.5
    n = 50
    nodeIDs = np.random.permutation(range(0, numNodes))[0:n]
    """
    infectionTimesList = []
    for nodeID in nodeIDs:
        _, infectionTimes = infect(nodes, p, nodeID, sorted_data)
        infectionTimesList.append(infectionTimes)

    pickle.dump(infectionTimesList, open("C:/Users/danie/Desktop/ProjectWork/templist3.txt","wb"))
    """
    infectionTimesList = pickle.load(open("C:/Users/danie/Desktop/ProjectWork/templist3.txt","rb"))
    nodeMeds = []
    for node in nodes:
        tempList = []
        for i in range(0, n):
            tempList.append(infectionTimesList[i][node])
        nodeMeds.append(np.median(tempList))
        print(tempList)
    
    """
    centralityMeasures = getData(network)    
    pickle.dump(centralityMeasures, open("C:/Users/danie/Desktop/ProjectWork/mylist.txt","wb"))
    """
    centralityMeasures = pickle.load(open("C:/Users/danie/Desktop/ProjectWork/mylist.txt","rb"))        
    
    fig = plt.figure()
    i = 0
    measureTitles = ['k-shell','Clustering coefficient','Degree','Strength','Betweenness centrality','Closeness centrality']
    print("Spearman rank-correlation coefficients")
    for measure, measureTitle in zip(centralityMeasures, measureTitles):
        i+=1
        ax = fig.add_subplot(3,2,i)
        ax.scatter(measure,nodeMeds)
        ax.set_xlabel(measureTitle)
        ax.set_ylabel('Median infection time')
        print(measureTitle + ":" + str(scipy.stats.spearmanr(measure, nodeMeds)[0]))
    plt.show()                
    
    #######
    # Task 5 
    ####### 
    
    numImmu = 10
    numSeed = 20
    immunizeInds = []
    for j in range(0, 6):
        immunizeInds.append([i[0] for i in sorted(enumerate(centralityMeasures[j]), key=lambda x:x[1])][-(numImmu+1):-1])
    immunizeInds.append(list(np.random.permutation(range(0, len(nodes)))[0:numImmu])) #random nodes
    neighbors = []
    while len(neighbors) < numImmu:
       focalNode = np.random.choice(range(0, numNodes))
       neighborNode = int(np.random.choice(network.neighbors(np.unicode(focalNode))))
       neighbors.append(neighborNode)
       neighbors = list(set(neighbors))
    immunizeInds.append(neighbors)
    allImInds = [item for sublist in immunizeInds for item in sublist]
    possibleSeedNodes = set(range(0, len(nodes))) - set(allImInds)
    seedNrs = list(np.random.permutation(list(possibleSeedNodes))[0:numSeed])

    p = 0.5
    n = 20
    """
    infectionTimesList = []
    for immunizeInd in immunizeInds:
        for seedNr in seedNrs:
            _, infectionTimes = ImmunizeInf(nodes, p, seedNr, sorted_data, immunizeInd)
            infectionTimesList.append(infectionTimes)

    pickle.dump(infectionTimesList, open("C:/Users/danie/Desktop/ProjectWork/mylist2.txt","wb"))
    """
    infectionTimesList = pickle.load(open("C:/Users/danie/Desktop/ProjectWork/mylist2.txt","rb"))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    timeStart = sorted_data[0][2]
    timeStop = np.sort(sorted_data, order='EndTime')[-1][3]
    timeRange = range(timeStart, timeStop, 1000)
    measureTitles = ['k-shell','Clustering coefficient','Degree','Strength','Betweenness centrality','Closeness centrality', 'Random', 'Social network']
    for i in range(0, len(immunizeInds)):
        fracInfected = []
        curInfectionTimes = []
        for k in range(i*numSeed, (i+1)*numSeed):
            curInfectionTimes.append(infectionTimesList[k].values())
        curInfectionTimes = [item for sublist in curInfectionTimes for item in sublist]
        curInfectionTimes = np.array(curInfectionTimes)
        numEl = len(curInfectionTimes)
        for j in timeRange:
            fracInfected.append(sum(curInfectionTimes<j)/numEl)
        ax.plot(timeRange, fracInfected, label=measureTitles[i])
    ax.legend(loc=4)
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction of infected airports')


    #######
    # Task 6 
    #######   

    p = 0.5
    n = 20
    nodeIDs = np.random.permutation(range(0, numNodes))[0:n]

    """
    infectionLinksList = []
    for nodeID in nodeIDs:
        _, _, infectionLinks = infectLinks(nodes, p, nodeID, sorted_data)
        infectionLinksList.append(infectionLinks)

    pickle.dump(infectionLinksList, open("C:/Users/danie/Desktop/ProjectWork/mylist3.txt","wb"))
    """
    
    infectionLinksList = pickle.load(open("C:/Users/danie/Desktop/ProjectWork/mylist3.txt","rb"))

    fractions = np.zeros((numNodes, numNodes))
    for i in range(0, numNodes):
            for k in range(0, n):
                if len(infectionLinksList[k][np.unicode(i)]) > 0:
                    fractions[i][int(infectionLinksList[k][np.unicode(i)])]+=1
    for i in range(0, numNodes):
        for j in range(i, numNodes):
            fractions[i][j] += fractions[j][i]
            fractions[i][j] = fractions[i][j]
    
    id_data = np.genfromtxt(
        'C:/Users/danie/Desktop/ProjectWork/US_airport_id_info.csv', delimiter=',', dtype=None, names=True)
    xycoords = {}
    for row in id_data:
        xycoords[str(row['id'])] = (row['xcoordviz'], row['ycoordviz'])
    edges = network.edges()
    numEdges = len(edges)
    weights = np.zeros(numEdges)
    for i in range (0, len(edges)):
        weights[i] = fractions[int(edges[i][0])][int(edges[i][1])]
    plot_network_usa(network, xycoords, weights/max(weights),edges, alpha = 1)
    
    netTemp = network.copy()
    edges = network.edges()
    for edge in edges:
        netTemp[edge[0]][edge[1]]['weight'] = -network[edge[0]][edge[1]]['weight']
    maxTree = nx.minimum_spanning_tree(netTemp, weight="weight")
    plot_network_usa(maxTree, xycoords, alpha=1)
    
    weights /= sum(weights) 
    edgeData = getEdges(network,edges)
    fig = plt.figure()
    measureTitles = ['Neighbourhood overlap','Weights','Betweenness centrality']
    for i in range(0, len(edgeData)):
        ax = fig.add_subplot(3,1,i+1)
        ax.scatter(edgeData[i], weights)
        ax.legend()
        ax.set_xlabel(measureTitles[i])
        ax.set_ylabel('Fraction of times infecting')
        ax.set_ylim(min(weights)-0.001, max(weights)+0.001)
        xlim = ax.get_xlim()
        if xlim[0] < 0:
                xlim =(-xlim[1] / 30, xlim[1])
        ax.set_xlim(xlim)
        print(measureTitles[i] + ":" + str(scipy.stats.spearmanr(weights, edgeData[i])[0]))
    plt.show()  