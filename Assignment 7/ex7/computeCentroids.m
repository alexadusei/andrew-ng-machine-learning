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

for k = 1:K
  % find a subset vector of all the values in idx that are of a certain class.
  % This will give you a vector of all the indexes of that class in idx.
  idx_subset_k = find(idx == k);

  % We'll use our subset of indices to map idx for that class to the values
  % of the examples in X. x_k represents all training examples with class k
  x_k = X(idx_subset_k, :);

  sigma_k = (1 / size(x_k, 1)) .* sum(x_k);
  centroids(k, :) = sigma_k;
end
% =============================================================


end

