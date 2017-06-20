function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% 'find' creates a vector of all indices in the dataset y that
% equate true to the boolean statement.
admitted = find(y == 1);
rejected = find(y == 0);

% plot the first test with the second test to create a point. Do this for all
% applicant data of those who were admitted vs those who were rejected
% the X matrix uses the first parameter as a vector, which is a vector of 
% indices the 2nd parameter looks at first 1st scores vs 2nd test scores
plot(X(admitted, 1), X(admitted, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(rejected, 1), X(rejected, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

% =========================================================================
hold off;

end
