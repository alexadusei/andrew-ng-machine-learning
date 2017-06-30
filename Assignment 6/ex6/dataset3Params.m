function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
lowest_prediction_error = inf;

% Iterate through all varations of C and all variations of sigma to find
% the lowest prediction error. The lowest prediction error shows the best
% combinations of C and sigma
for ci = 1:length(C_vals)
  C_current = C_vals(ci);

  for si = 1:length(sigma_vals)
    sigma_current = sigma_vals(si);

    model = svmTrain(X, y, C_current, @(x1, x2) gaussianKernel(x1, x2, sigma_current));
    predictions = svmPredict(model, Xval);
    current_prediction_error = mean(double(predictions ~= yval));

    if current_prediction_error < lowest_prediction_error
      lowest_prediction_error = current_prediction_error;
      C = C_current;
      sigma = sigma_current;
    end
  end
end
% =========================================================================

end
