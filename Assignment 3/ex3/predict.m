function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% What the fuck is any of this though???

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Produce predictions for all the neurons in the hidden layer, using the 
% Sigmoid function (remember, this will be the same process as logistic reg.)
% Remember to add a bias unit to our hidden layer, which we'll be using for our
% next predictions with the next layer
hidden_layer_probabilities = [ones(m, 1) sigmoid(X * Theta1')];

% Next, compute predictions for the neurons of the next layer, which is the 
% output layer. This will be a 10-vectory classifer of probabilities for each
% class (0 - 9)
output_layer_probabilities = sigmoid(hidden_layer_probabilities * Theta2');

% For each example, get the classifier with the highest certainty.
% Do this for all examples (5000)
[values, indices] = max(output_layer_probabilities, [], 2);

p = indices;
% =========================================================================


end
