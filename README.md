# Clustering
An implentation of the K-means++ and K-medians clustering algorithms, and comparison of B-cubed metrics for different values of k

## Objective
The aim is to cluster objects in the dataset into k many clusters, where each cluster has a represenative around whihc it is centered. K-means++ and K-medians iteratively minimise the sum of distances between objects in the dataset and the represenative of the cluster they are assigned to.

The code runs these algorithms on the dataset for values of K in the range 1 to 9, computing B-cubed F-score, precision and recall for each value of K to find the optimal value of k. Since objects in the dataset are split into 4 categories, the expected result is that k=4 is optimal. 

## Implementation
### K-means and K-medians
Each iteration of the algorithms consist of assigning datapoints to representatives, evaluating the resulting clustering and choosing new representatives for the next iteration.

For K-means Euclidean distance is used as the distance measure, and for K-medians Manhattan distance.

1) assign: each datapoint is assigned to the representative that is closest to it. The clustering is saved in case the next iteration's clustering is inferior. 
2) evaluate: the sum of distances between datapoints and the representative of their cluster is computed. If the value of this sum is larger than the result of the evaluation step of the previous iteration we return the previous iteration's clustering. 
3) choose new representatives: for each cluster we compute the coordinate-wise mean (or median) of objects in the cluster. The mean (or median) 'object' created is used as a representative for the assign step of the next iteration. 

Exit condition: the new representatives fail to improve on the results of the previous iteration, or user defined max iterations limit is reached

### B-Cubed Metrics
B-cubed metrics for clustering require ground truth categories for each object, which are avilable in this case. These metrics are computed per object and then averaged. 

B-cubed precision(x) = # objects in x's cluster that are in the same ground truth category as x / # objects in x's cluster
B-cubed recall(x) = # objects in x's cluster that are in the same ground truth category as x / # objects in x's ground truth category

B-cubed F-score(x) = 2 * (precision(x) * recall(x)) / precision(x) + recall(x)

For all of these metrics, the higher the value the better the clustering. Note that for k=1 recall will be perfect. 

### Initial representative selection: the ++ in k-means++
Selecting initial represenatives randomly can result in 'lucky' or 'unlucky' initialisations. K-means++ uses a system for selecting representatives that ensures that the initial representatives are optimally spaced out relative to the dataset. 

## Results
![Figure_1](https://user-images.githubusercontent.com/34168073/185670576-447d2d72-5d43-4ad2-8344-e6107497c918.png)
![Figure_2](https://user-images.githubusercontent.com/34168073/185670521-c06014f9-803c-4cf3-a17d-56162624fcb2.png)
![Figure_3](https://user-images.githubusercontent.com/34168073/185670547-e9c75350-be49-4691-91b1-0c488bcaf663.png)
![Figure_4](https://user-images.githubusercontent.com/34168073/185670564-47e9249d-b12f-4dfb-9b8b-9d25a07d5c4a.png)

## Dataset
The dataset consists of categorised words paired with a vector of numerical values is small scale and have been uploaded to this repository.
