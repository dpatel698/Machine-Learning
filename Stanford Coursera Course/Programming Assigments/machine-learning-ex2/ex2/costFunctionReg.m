function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h0 = sigmoid(X * theta);
reg = (lambda ./(2 .* m)) * (theta'(:,2:length(theta')) * 
theta(2:length(theta),:));  % The regularization value

J = ((-1/m) .* sum((y .* log(h0)) + ((1 - y) .* log(1 - h0)))) + reg;
J = sum(J)/length(J);


grad0 = sum((h0 - y) .* X) ./ m;

grad = (sum((h0 - y) .* X) ./ m) + (lambda./m).* theta'

grad = [grad0(1,1) , grad(:,2:end)];



% =============================================================

end
