function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1.1: Forward Propagation
% ----------------------------

% Add bias unit to the X matrix
X = [ones(m, 1) X]; % 5000 x 401

% approximate the hidden layer and add bias unit
hidden_layer_probs = [ones(m, 1) sigmoid(X * Theta1')]; % 5000 x 26

% approximate the output layer
output_layer_probs = [sigmoid(hidden_layer_probs * Theta2')]; % 5000 x 10

% Part 1.2.1: NN Cost Function (Unregularized)
% --------------------------------------------

% Previously, we tweaked our NN to just be 1-dimensional vectors for y and h,
% soley because we were seeing their values as classifiers from 1-10. Now,
% we're representing our y and h the same way, but tagging them as m x 10 matrices,
% where each trainig example is a 1 x 10 horizontal vector, and each index
% represents the class from 0-10. The correct class will have a '1' for that
% respective index, while the rest are 0s
% ex: A number three will be [0 0 1 0 0 0 0 0 0 0]

% Because we're representing y and h are matrices, they will be Y and H respectively,
% with sizes (m x k).
Y = zeros(m, num_labels); % 5000 x 10
H = output_layer_probs; % 5000 x 10

for i = 1:m
  % for training example i of Y, go to training example i of y, which will just
  % be a value from 1-10. Use this index to go to row i, training example y(i)
  % of Y, and turn only that value into a 1, while everything else is a 0.
  % This essentially converts our vector values into a matrix of 0s and 1s
  Y(i, y(i)) = 1;
end

% Because Y and H are matrices now, we can't use the cost function the same
% way we did before, as that was vector multiplication, which returns a scalar
% value. Matrix multiplication returns a new matrix. All we have to do here is
% get the sums of the diagonals.

% Why do we do this? Because we're taking all training examples of Y with
% respect to the output being k, and multiplying it by all training examples of
% H with respect to the output being k. In other words (let's see how we did
% with our hypothesis H in comparison to the correct answer Y, for each class.
% So check the average with class 1, then class 2, then class 3... etc).
% We do this with Y' (10 x 5000) and H (5000 x 10), giving us a (k x k) (or 10x10
% in this case) matrix. This means that for each row, there is only one
% desirable output, and that is the kth row and kth column value. If we are
% looking for how we did predicting a 4, then we'd have to look at row 4, col 4
% only. Again, this: "look at the '1' class column-vector for H and the '1' class
% row-vector for Y'. Vector-multiply these so we get the average of how many times
% Hk got the same answer as Yk. Put the result in (Y'*H) in (1x1). Do the same for
% class 2, and put it in (Y'*H) (2x2), and 3 for (3x3) etc...". This makes only
% the diagonol values in the resulting matrix important.

% trace() returns the sum of only the diagonal values in a matrix (top-left to
% bottom-right

% Why is H left alone and not converted into the classes 1-10 like we did in
% assignment 4? Because H is our hypothesis. We did that early just as a
% convenience, but you want to leave your hypotheses as probabilities.
% H will have training examples of 10-dimensional vectors, where each index
% has a probability of being that class. If your hypothesis is strong, you'll
% have one high probability for a class while all the rest are low probabilities
% E.x: "I believe this drawing is a 9, with 0.923 certainty. The other numbers are
% low with 0.0023 certainty of being them"
J = (1 / m) * (trace(-Y' * log(H)) - trace((1 - Y)' * log(1 - H)));


% Part 1.2.2: NN Cost Function (Regularized)
% ------------------------------------------

% Just use the previous calculation of J and add the regularized product to it
% Remember to remove the bias units
J = J + (lambda / (2 *m )) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));

% Part 2: Backpropagation
% ----------------------------

  % Part 1:
  % Set a(t) = x(t) and perform forwad propagation.
  % Don't forget to add bias unit with [1 <vector>]
  a1 = X; % X already has bias unit from earlier use (5000 x 401)

  z2 = a1 * Theta1'; % 5000 x 25
  a2 = [ones(m, 1) sigmoid(z2)]; % 5000 x 26
  z3 = a2 * Theta2'; % 5000 x 10
  a3 = sigmoid(z3); % 5000 x 10

  % Part 2:
  % For each output in the final layer, find the margin of error with
  % the variable delta, our "error term"

  D3 = a3 - Y; % (5000 x 10)

  % Part 3:
  % Set the remaining deltas per layer with their respective calculations
  % first product is (5000 x 25), second product is (5000 x 25)
  D2 = (D3 * Theta2(:, 2:end)) .* sigmoidGradient(z2); % (5000 x 25)

  % Part 4:
  % Accumulate the gradient from this example using

  % (25 x 5000) * (5000 x 401)
  Delta1 = D2' * a1; % (25 x 401)

  % (10 x 5000) * (5000 x 26)
  Delta2 = D3' * a2; %(10 x 26)

  Theta1_grad = (1 / m) * Delta1;
  Theta2_grad = (1 / m) * Delta2;

% Part 3: Backpropagation with Regularization
% -------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
