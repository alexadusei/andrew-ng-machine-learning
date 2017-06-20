% Parameters
% -------------
% X = a matrix
% y = a vector
% theta = a scalar value
function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% * = matrix multiplication
% .* = multiply every element in matrices

% X is a 97x2 matrix. Theta is a 2x1 matrix. The matrix orientation is different
% in MATLAB than it is in theory, so transpositions will vary from the formulas
% Andrew Ng provides.. Focus less on formula transposition and more on matrix
% properties (matrix A column must == matrix B row for matrix multiplication)
% (92x2) * (2x1) results in a (97x1) matrix

% OTHER NOTE: Technically, this is correct. X * theta includes all training 
% examples, but if we just looked at ONE training example, then x = [x0 x1]
% is already transposed as x'. This is equivalent to theta' * x. Andrew Ng
% just adding some variability to throw you off. Stay sharp
h = X * theta;
J = (1/(2 * m)) * sum((h - y) .^ 2);

% =========================================================================

end
