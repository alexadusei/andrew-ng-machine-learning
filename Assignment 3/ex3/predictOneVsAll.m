function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.

% Use the sigmoid function with the training examples & all the learned thetas
% to get a probability of the example being each of the classifiers (10).
% For one example, you'd get a 10-lengthed vector with a probability for
% each class. E.g: "I'm 13% sure its a '1', 23% sure it's a '2', 97% sure it's
% a '3', etc". You'd do this for each example (5000 examples in this case).
% This gives us a 10 x 5000 matrix of probabilities, where each column
% represents the certainty for each classifier.
probabilities = sigmoid(X * all_theta');

% use max function to find maximum value of each classifier.
% max(X, [], 1) will find the largest value per column (VERTICAL MAX)
% max(X, [], 2) will find the largest value per row (HORIZONTAL MAX)
[values, indices] = max(probabilities, [], 2);

p = indices;
% =========================================================================


end
