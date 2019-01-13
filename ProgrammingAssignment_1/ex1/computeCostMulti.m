function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
disp(theta);

coef = 1/(2 * m);

matrix_hasil_h_theta = X *  theta;
matrix_hasil_kurangi_y = matrix_hasil_h_theta - y;
matrix_hasil_kurangi_y_kuadrat = matrix_hasil_kurangi_y .^ 2;
sum_of_matrix_hasil_kurangi_y_kuadrat = sum(matrix_hasil_kurangi_y_kuadrat);

% You need to return the following variables correctly 
J = coef * sum_of_matrix_hasil_kurangi_y_kuadrat;




% =========================================================================

end
