function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%


% Vectorized Centroid Computation:

% For vectorization we need for each cluster:
% The X data points in that cluster (also the number of data points)
% Compute the mean for each unique cluster and assign that as the centroid value

% First we make a matrix of size n x k whose columns will be filtered for
% containing the index of each cluster [1...K] the columns will multiply with 
% X to sum only the training examples that correspond to that cluster
clusters = 1:K;
filter_matrix = ones(m, K) .* idx;
filter_matrix = (filter_matrix == clusters); % m x K
cluster_sizes = (sum(filter_matrix) .^ -1); % 1 x K

centroids = ((X' * filter_matrix) .* cluster_sizes)' ; % K x n








% =============================================================


end

