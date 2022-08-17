import numpy as np
import matplotlib.pyplot as plt
import time

# Unpacks the dataset into a dictionary
# Parameters: files - the files to be unpacked
# Return: dictionary - 'instance': (np.array([itemFeatures]), trueClassLabel)
def unpackData(*files):
    data = dict()
    for file in files:
        with open(file, 'r') as f: 
            lines = f.readlines()
            for line in lines: 
                line = line.split()
                # The feature values are stored as strings so they need to be converted to floats for future computation
                data[line[0]] = (np.array([float(x) for x in line[1:]]), file)
    return data

data = unpackData('animals', 'countries', 'fruits', 'veggies')

# Counts for each label type for computation of b-cubed recall
labelCounts = {'animals': 50, 'countries': 161, 'fruits': 58, 'veggies': 58}

# Parameters: x, y, ndarrays
# Returns: Euclidean distance between x and y, float
def eucDist(x, y):
    return np.linalg.norm(x-y) 

# Parameters: x, y, ndarrays
# Returns: Manhattan distance between x and y, float
def manDist(x, y):
    return np.abs(x - y).sum()

# Selects initial representatives according to the k-means++ selection scheme
# A random initial representative is selected form the dataset
# Subsequent representatives are selected with a probability proportional to their distance to the most recently chosen representative
# This function adapts the k-mean++ selection scheme for use with k-medians, with Manhattan distance used in place of Euclidean
# Parameters: k, int - number of clusters; dist, function - must be either eucDist or manDist, default is eucDist
# Returns: list containing feature vectors of the chosen initial representatives
def initialiseReps(k, dist=eucDist):
    representatives = []
    for _ in range(k):
        # Random seed here is used to ensure reproducibility of results
        np.random.seed(18)

        # The first representative is chosen randomly from the dataset
        if representatives == []:
            choice = np.random.choice(list(data.keys()))
            representatives.append(data[choice][0])
        else:
            # Computes distance to most recently assigned representative using chosen distance measure
            distToLatestRep = {instance: dist(data[instance][0], representatives[-1]) for instance in data.keys()}
            # Computes sum of distances between datapoints and most recently assigned representative 
            sumDistToRep = np.array([value for value in distToLatestRep.values()]).sum()
            # Computes probabilities of selection for each datapoint 
            selectionProbabilities = {instance: distToLatestRep[instance]/sumDistToRep for instance in distToLatestRep.keys()}

            # Sets lower and upper bounds for the selection range of each datapoint
            selectionRanges = {key: (0 if i == 0 else sum(list(selectionProbabilities.values())[:i]), v if i == 0 else (sum(list(selectionProbabilities.values())[:i])+v)) for (i, (key, v)) in enumerate(selectionProbabilities.items())}

            # Random seed here is used to ensure reproducibility of results
            np.random.seed(18)
            selection = np.random.random()

            # Checks which selection range the randomly generated values falls in and makes the instance corresponding to that selection range a representative
            for instance in selectionRanges:
                if selection >= selectionRanges[instance][0] and selection < selectionRanges[instance][1]:
                    representatives.append(data[instance][0])
    return representatives

# Finds the index of the closest respresentative to an instance
# Parameters: instance, tuple - (feature vector stored as an ndarray, true label), representatives, list - contains k ndarrays, dist, function - the distance measure being used, eucDist or manDist
# Returns: closest rep, int - the index of the closest representative (which corresponds to the clusterID of that representative's cluster)
def closestRep(instance, representatives, dist=eucDist):
    return np.argmin([dist(data[instance][0], rep) for rep in representatives])

# Assigns objects to clustes based on their closest representative
# Parameters: representatives, list - contains k ndarrays, dist, function - the distance measure being used, eucDist or manDist
# Returns: clustering, dictionary - {clusterID: [list of feature vectors of objecs assigned to a particular cluster]}
def assign(representatives, dist=eucDist):
    clusters = {i: [] for i in range(len(representatives))}
    for instance in data.keys():
        clusters[closestRep(instance, representatives, dist)].append(instance)

    return clusters

# Computes sum of distance between objects and their closest representative
# Used to determine a stopping point for K-means and K-medians clustering
# Parameters: representatives, list - contains k ndarrays, dist, function - the distance measure being used, eucDist or manDist
# Returns: sum of distances between obejcts and closest reps, float
def objectiveFunction(representatives, dist=eucDist):
    totalDist = 0

    for (id, cluster) in assign(representatives, dist=dist).items():
        totalDist += np.array([dist(data[instance][0], representatives[id]) for instance in cluster]).sum()

    return totalDist

# Implements the k-means++ clustering algorithm
# Parameters: k, int - the number of clusters; maxIter, int - maximum iterations before stopping
# Returns: clustering, dictionary: {clusterID, int: [list of objects, strings]}
def kMeans(k, maxIter):
    reps = initialiseReps(k)

    for iteration in range(maxIter):
        # Assign step
        clustering = assign(reps)
        # Optimisation step
        newReps = []
        for (id, cluster) in clustering.items():
            if len(cluster) == 0:
                # if a cluster has no objects assigned to it during the assign step, no new representative is assigned
                newReps.append(reps[id])
                print(f"Empty cluster detected in iteration {iteration}, its representative will be preserved for the next assign step")
            else:
                newReps.append(np.array([data[instance][0] for instance in cluster]).mean(axis=0))
   
        # Check whther stopping condition met
        if objectiveFunction(newReps) >= objectiveFunction(reps):
            return clustering
        elif iteration == maxIter-1:
            return clustering
        else: 
            reps = newReps
            continue

# Implements the k-medians clustering algorithm
# The scheme used for selecting initial representatives is an adpted version of k-means++ initial rep selection
# The difference is that manhattan distance is used here to deterine the probability that an object will be selected as the next representative in place of Euclidean distance 
# Parameters: k, int - the number of clusters; maxIter, int - maximum iterations before stopping
# Returns: clustering, dictionary: {clusterID, int: [list of objects, strings]}
def kMedians(k, maxIter):
    reps = initialiseReps(k, dist=manDist)
    
    for iteration in range(maxIter):
        # Assign step
        clustering = assign(reps, dist=manDist)
        # Optimisation step
        newReps = []
        for (id, cluster) in clustering.items():
            if len(cluster) == 0: 
                # if a cluster has no objects assigned to it during the assign step, no new representative is assigned
                newReps.append(reps[id])
                print(f"Empty cluster detected in iteration {iteration}, its representative will be preserved for the next assign step")
            else:
                newReps.append(np.median(np.array([data[instance][0] for instance in cluster]), axis=0))
   
        # Check whther stopping condition met
        if objectiveFunction(newReps, dist=manDist) >= objectiveFunction(reps, dist=manDist):
            return clustering
        elif iteration == maxIter-1:
            return clustering
        else: 
            reps = newReps
            continue

# Computes B-cubed precision, recall and F-score for each object in the dataset and returns the average of these to give the B-Cubed scores for a given clustering
# Parameters: clustering, dictionary: {clusterID, int: [list of objects, strings]}
# Returns: B-cubed scores, floats - B-cubed precision, recall and F-score
def bCubedScore(clustering):
    prec = []
    recall = []
    fScore = []

    for (id, cluster) in clustering.items():
        for instance in cluster:
            p = sum([1 if data[instance][1] == data[otherInstance][1] else 0 for otherInstance in cluster])/ len(clustering[id])
            r = sum([1 if data[instance][1] == data[otherInstance][1] else 0 for otherInstance in cluster])/ labelCounts[data[instance][1]]
            f = (2*(p*r))/(p+r)

            prec.append(p)
            recall.append(r)
            fScore.append(f)
            #if (f < p and f < r) or (f > p and f > r):
                #print(f"scores for {instance} are spurious!\nprec = {p}\trecall = {r}\tF-score = {f}")

    bCubedPrec = np.array(prec).mean()
    bCubedRecall = np.array(recall).mean()
    bCubedFScore = np.array(fScore).mean()

    return bCubedPrec, bCubedRecall, bCubedFScore

### TASK 3: B-cubed scores for K-means ###
scores = dict()
for i in range(1, 10):
    scores[i] = bCubedScore(kMeans(i, 100))

print(f"K-means scores: {scores}")

plt.subplots()
plt.plot(np.arange(1, 10), np.array([score[0] for score in scores.values()]), color='blue', label='Precision')
plt.plot(np.arange(1, 10), np.array([score[1] for score in scores.values()]), color='orange', label='Recall')
plt.plot(np.arange(1, 10), np.array([score[2] for score in scores.values()]), color='green', label='F-Score')
plt.xlabel('Values of K')
plt.legend(loc=4)
plt.title('B-Cubed performance measures of K-means clustering')
plt.show()

### TASK 5: B-cubed scores for K-medians ###
scores = dict()
for i in range(1, 10):
    scores[i] = bCubedScore(kMedians(i, 100))

print(f"K-medians scores: {scores}")

plt.subplots()
plt.plot(np.arange(1, 10), np.array([score[0] for score in scores.values()]), color='blue', label='Precision')
plt.plot(np.arange(1, 10), np.array([score[1] for score in scores.values()]), color='orange', label='Recall')
plt.plot(np.arange(1, 10), np.array([score[2] for score in scores.values()]), color='green', label='F-Score')
plt.xlabel('Values of K')
plt.legend(loc=4)
plt.title('B-Cubed performance measures of K-medians clustering')
plt.show()

### Normalising dataset to L2 unit length ###
def l2Unit(vector):
    return vector*1/np.linalg.norm(vector)

for instance in data.keys():
    data[instance] = (l2Unit(data[instance][0]), data[instance][1])

### TASK 4: B-cubed scores for K-means on dataset normalised to L2 unit vector length ###
scores = dict()
for i in range(1, 10):
    scores[i] = bCubedScore(kMeans(i, 100))

print(f"K-means scores, with L2 Unit normalisation: {scores}")

plt.subplots()
plt.plot(np.arange(1, 10), np.array([score[0] for score in scores.values()]), color='blue', label='Precision')
plt.plot(np.arange(1, 10), np.array([score[1] for score in scores.values()]), color='orange', label='Recall')
plt.plot(np.arange(1, 10), np.array([score[2] for score in scores.values()]), color='green', label='F-Score')
plt.xlabel('Values of K')
plt.legend(loc=4)
plt.title('B-Cubed performance of K-means with data normalised to L2 unit length')
plt.show()

### TASK 6: B-cubed scores for K-means on dataset normalised to L2 unit vector length ###
scores = dict()
for i in range(1, 10):
    scores[i] = bCubedScore(kMedians(i, 100))

print(f"K-medians scores, with L2 Unit normalisation: {scores}")

plt.subplots()
plt.plot(np.arange(1, 10), np.array([score[0] for score in scores.values()]), color='blue', label='Precision')
plt.plot(np.arange(1, 10), np.array([score[1] for score in scores.values()]), color='orange', label='Recall')
plt.plot(np.arange(1, 10), np.array([score[2] for score in scores.values()]), color='green', label='F-Score')
plt.xlabel('Values of K')
plt.legend(loc=4)
plt.title('B-Cubed performance of K-medians with data normalised to L2 unit length')
plt.show()