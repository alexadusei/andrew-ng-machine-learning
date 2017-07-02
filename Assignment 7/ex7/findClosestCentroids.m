function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
m = size(X, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


% go through each training example
for i = 1:m
  smallest_dist = inf;
  cluster_k = 0;

  % go through each cluster class k
  for k = 1:K
    % Remember: |v| follows Pythagorean's Thereom: sqrt(v1^2 + v2^2 + ... + vn^2)
    x_diff = X(i, :) - centroids(k, :);
    x_diff_length = sqrt(sum(x_diff .^ 2)) ^ 2;

    % keep iterating through the smallest distance and its associated cluster
    if x_diff_length < smallest_dist
      smallest_dist = x_diff_length;
      cluster_k = k;
    end
  end

  % after finding the smallest difference for the clusters, assign them to c(i)
  idx(i) = cluster_k;
end
% =============================================================

end


