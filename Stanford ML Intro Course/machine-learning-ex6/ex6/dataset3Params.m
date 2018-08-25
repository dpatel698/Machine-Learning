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

% This vector contains the values to try for C and sigma
vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

% Next we will make a matrix of all possible C and sigma pairs from vals
[test_c test_sigma] = meshgrid(vals, vals);
pairs = [test_c(:) test_sigma(:)];

% Next we will train the model on the cross validation set and use that to make
% make predictions and calculate the error for each C and sigma pair used
% to generate the model that made the predictions
errors = zeros(length(pairs), 1);
for i = 1:length(pairs)
  test_c = pairs(i, 1);
  test_sigma = pairs(i, 2);
  model = svmTrain(X, y, test_c, @(x1, x2) gaussianKernel(x1, x2, test_sigma));
  % Now we make predictions on the current model and generate the prediction
  % error for the current C and sigma 
  predictions = svmPredict(model, Xval);
  errors(i) = mean(double(predictions ~= yval));

endfor

% Now we find the index of the minimum error and use the C and sigma at that 
% index as out optimal C and sigma
[val, low_index] = min(errors);
C = pairs(low_index, 1);
sigma = pairs(low_index, 2);







% =========================================================================

end
