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
  Activation_layer_1 = X; % X already has bias unit from earlier use (5000 x 401)

  Z_layer_2 = Activation_layer_1 * Theta1'; % 5000 x 25
  Activation_layer_2 = [ones(m, 1) sigmoid(Z_layer_2)]; % 5000 x 26
  Z_layer_3 = Activation_layer_2 * Theta2'; % 5000 x 10
  Activation_layer_3 = sigmoid(Z_layer_3); % 5000 x 10

  % Part 2:
  % For each output in the final layer, find the margin of error with
  % the variable 'delta', our "error term"

  % See how bad we did for the output layer. Just do a difference of values from
  % our output to our answers (y)
  Delta_layer_3 = Activation_layer_3 - Y; % (5000 x 10)

  % Part 3:
  % See how bad we did for our hidden layers. We can't just compare these to
  % our Y value because they're not based on the final output. We'll have to
  % "backtrack" to figure out our margin of error for our hidden layers. A good
  % way of thinking about this is with this scenario: "We see our margin of error
  % for our output layers by comparing their answers to our answer-book 'y'. We
  % get the differences here. If the differences are extremely small, then our margin
  % of error for our output layer is small. If we have some neurons that have
  % a large margin of error, then we messed up! The only person to blame right
  % now is the previous neuron(s) that gave this bad neuron its bad answer. We
  % must penalize them"
  % "In order to penalize them, we let these previous neurons know how bad
  % THEY did by showing them the margin of error that we got based on their
  % inputs to us ('us' being the output layer. We share with them how bad we
  % did by giving them their weights multiplied by our margin of error, and multiply
  % this by the derivative sigmoid function". This process goes on until we
  % get to the last hidden layer, which is going to be Layer #2.

  % Set the remaining deltas per layer with their respective calculations
  % first product is (5000 x 25), second product is (5000 x 25)
  Delta_layer_2 = (Delta_layer_3 * Theta2(:, 2:end)) .* sigmoidGradient(Z_layer_2); % (5000 x 25)

  % Part 4:
  % Now we accumulate our margins of error into one big vector for each layer.
  % we cann this the delta accumulator. For each layer, it gets our calculated
  % margin of error vector for said layer and multiplies it by our activation
  % layer. We start our accumulator layer in layer 1, and proceed until we get
  % to to second last layer (we're not including the output layer here), so that
  % is layer L (in this case, that's just layer 1 and layer 2).

  % (25 x 5000) * (5000 x 401)
  % This will be size (25 x 401), as in, it is an average of all the training
  % examples for each neuron unit (25) with each pixel (401). We'll balance out
  % this average by multiplying it by (1/m) later on
  Delta_accumulator_layer_1 = Delta_layer_2' * Activation_layer_1; % (25 x 401)

  % (10 x 5000) * (5000 x 26)
  Delta_accumulator_layer_2 = Delta_layer_3' * Activation_layer_2; %(10 x 26)

  % Once we get these two delta accumlator layers, we average it out with all of
  % our training examples. This becomes the derivative of J with respect to theta
  % for each layer (think of J derivatives per layer). The derivative of J
  % is D, which is (1/m) * delta_accumulator, which is delta * a, where delta
  % is (previous_delta * theta) .* g', until our last 'previous_delta' is
  % (a - y) [this is a pure literary description of what's going on. See
  % notes taken to get a better understanding. It's looking at all the steps
  % for backpropagation in reverse].

  % We rename these as 'Theta1_grad' and 'Theta2_grad'. As always, for
  % regularization, we don't apply the bias layers, so we don't regularize
  % j = 0 (or j = 1 in MATLAB's case), while regularizing the rest of the indices
  Theta1_grad(:, 1) = (1 / m) * Delta_accumulator_layer_1(:, 1);
  Theta1_grad(:, 2:end) = ((1 / m) * Delta_accumulator_layer_1(:, 2:end)) + ((lambda / m) * (Theta1(:, 2:end)));

  Theta2_grad(:, 1) = (1 / m) * Delta_accumulator_layer_2(:, 1);
  Theta2_grad(:, 2:end) = ((1 / m) * Delta_accumulator_layer_2(:, 2:end)) + ((lambda / m) * (Theta2(:, 2:end)));
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
