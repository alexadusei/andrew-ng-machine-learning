function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

x_diff = x1 - x2;

% Remember: |v| follows Pythagorean's Thereom: sqrt(v1^2 + v2^2 + ... + vn^2)
x_diff_length = sqrt(sum(x_diff .^ 2));

sim = exp(-(abs(x_diff_length) ^ 2) / (2 * (sigma ^ 2)));
% =============================================================
    
end
