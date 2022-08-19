# Clustering
An implentation of the K-means++ and K-medians clustering algorithms, and comparison of B-cubed metrics for different values of k

## Objective
The aim is to cluster objects in the dataset into k many clusters, where each cluster has a represenative around whihc it is centered. K-means++ and K-medians iteratively minimise the sum of distances between objects in the dataset and the represenative of the cluster they are assigned to.

The code runs these algorithms on the dataset for values of K in the range 1 to 9, computing B-cubed F-score, precision and recall for each value of K to find the optimal value of k. Since objects in the dataset are split into 4 categories, the expectated result is that k=4 is optimal. 

## Implementation
### Initial representative selection: the ++ in k-means++

## Results
![Figure_1](https://user-images.githubusercontent.com/34168073/185670576-447d2d72-5d43-4ad2-8344-e6107497c918.png)
![Figure_2](https://user-images.githubusercontent.com/34168073/185670521-c06014f9-803c-4cf3-a17d-56162624fcb2.png)
![Figure_3](https://user-images.githubusercontent.com/34168073/185670547-e9c75350-be49-4691-91b1-0c488bcaf663.png)
![Figure_4](https://user-images.githubusercontent.com/34168073/185670564-47e9249d-b12f-4dfb-9b8b-9d25a07d5c4a.png)

## Dataset
The dataset consists of words paired with a vector of numerical values is small scale and have been uploaded to this repository.
