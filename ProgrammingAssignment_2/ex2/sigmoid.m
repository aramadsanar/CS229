function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
  g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

  z_minus = -z;
  
  e_pangkat_z_minus_tambah_satu = 1 + (e.^ z_minus);
  
  e_pangkat_z_minus_tambah_satu_pangkat_min_satu  = e_pangkat_z_minus_tambah_satu .^ -1;

  g = e_pangkat_z_minus_tambah_satu_pangkat_min_satu;

% =============================================================

end
