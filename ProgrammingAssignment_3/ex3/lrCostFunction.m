function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

  theta_pertama = theta(1);
% (lambda/2m) * sum(theta_dari_2 .^ 2)
  theta_dari_2 = theta(2:end, :);
  theta_pasang_lagi = [0;theta_dari_2];
  koef_lambda_dibagi_2m = lambda /(2* m);
  sum_row_2_kebawah_pangkat_2 = sum(theta_pasang_lagi .^ 2);
  
  regularization_value_J = koef_lambda_dibagi_2m * sum_row_2_kebawah_pangkat_2;
  
% (lambda/m) * theta_dari_2  
  koef_lambda_dibagi_m = lambda/m;
  regularization_value_grad = koef_lambda_dibagi_m * theta_dari_2;
  
  %sprintf('theta pertama = %d\n', theta_pertama);
  regularization_value_grad = vertcat([theta_pertama], regularization_value_grad);
  
  koef = 1/m;
  %theta_t = theta';
  base_hypothesis_value = X * theta;
  
  sigmoid_base_hypothesis_value = sigmoid(base_hypothesis_value);
  
  if_1 = y' * log(sigmoid_base_hypothesis_value);
  if_0 = (1-y)' * log(1 - sigmoid_base_hypothesis_value);
  
  theta_dua_kebawah = theta(2:size(theta));
  theta_reg = [0;theta_dua_kebawah];
  
  
  if_combined = -1 * (if_1 + if_0);
  
  sum_of_if_combined = sum(if_combined);
  
  J_pertama = J(1,:);
  J_pertama = (koef * sum_of_if_combined);
  potongan_J_pertama = J_pertama(1,:);
  
  %regularization_value_J = koef_lambda_dibagi_2m * theta_reg' * theta_reg;
  
  J = (koef * sum_of_if_combined) + regularization_value_J;
  %J(1) = potongan_J_pertama; 
   %J = J;
  grad_pertama = grad(1,:);
  
  grad_pertama = (koef * (X' * (sigmoid_base_hypothesis_value - y)));
  
  potongan_grad_pertama = grad_pertama(1,:);
  grad = (koef * (X' * (sigmoid_base_hypothesis_value - y))) + regularization_value_grad;

  grad(1) = potongan_grad_pertama;
  grad = grad;







% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%










% =============================================================

grad = grad(:);

end
