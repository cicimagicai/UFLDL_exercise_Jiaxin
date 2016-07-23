clear all;

addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled

% Load the MNIST data for this exercise.
% train.X and test.X will contain the training and testing images.
%   Each matrix has size [n,m] where:
%      m is the number of examples.
%      n is the number of pixels in each image.
% train.y and test.y will contain the corresponding labels (0 or 1).
binary_digits = true;
[train,test] = ex1_load_mnist(binary_digits);

% Add row of 1s to the dataset to act as an intercept term.
train.X = [ones(1,size(train.X,2)); train.X]; 
test.X = [ones(1,size(test.X,2)); test.X];

% Training set dimensions
m=size(train.X,2);
n=size(train.X,1);

% Train logistic regression classifier using minFunc
options = struct('MaxIter', 100);

% First, we initialize theta to some small random values.
%theta1 = rand(n,1)*0.001;

% Call minFunc with the logistic_regression.m file as the objective function.
%
% TODO:  Implement batch logistic regression in the logistic_regression.m file!
%
%tic;
%theta1=minFunc(@logistic_regression, theta1, options, train.X, train.y);
%fprintf('Optimization took %f seconds.\n', toc);

% Now, call minFunc again with logistic_regression_vec.m as objective.
%
% TODO:  Implement batch logistic regression in logistic_regression_vec.m using
% MATLAB's vectorization features to speed up your code.  Compare the running
% time for your logistic_regression.m and logistic_regression_vec.m implementations.
%
% Uncomment the lines below to run your vectorized code.
theta2 = rand(n,1)*0.001;

tic;
theta2=minFunc(@logistic_regression_vec, theta2, options, train.X, train.y);
fprintf('Optimization took %f seconds.\n', toc);

% Print out training accuracy.
tic;
train_accuracy2 = binary_classifier_accuracy(theta2,train.X,train.y);
fprintf('Training accuracy: %2.1f%%\n', 100*train_accuracy2);

% Print out accuracy on the test set.
test_accuracy2 = binary_classifier_accuracy(theta2,test.X,test.y);
fprintf('Test accuracy: %2.1f%%\n', 100*test_accuracy2);

% Gradient Check  
average_error2 = grad_check(@logistic_regression_vec,theta2,200,train.X,train.y);  
fprintf('Average error :%f\n',average_error2);  

% Plot predicted prices and actual prices from training set.

actual_prices = test.y;
predicted_prices2 = sigmoid(theta2'*test.X);

% Plot predictions on test data.
plot_prices=true;
if (plot_prices)
  [actual_prices,I] = sort(actual_prices);
  predicted_prices2=predicted_prices2(I);
  figure;
  plot(actual_prices, 'rx');
  hold on;
  plot(predicted_prices2,'bx');
  legend('Actual Price', 'Predicted Price');
  xlabel('House #');
  ylabel('House price ($1000s)');
end


