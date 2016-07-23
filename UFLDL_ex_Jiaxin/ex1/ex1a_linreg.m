%
%This exercise uses a data from the UCI repository:
% Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
% http://archive.ics.uci.edu/ml
% Irvine, CA: University of California, School of Information and Computer Science.
%
%Data created by:
% Harrison, D. and Rubinfeld, D.L.
% ''Hedonic prices and the demand for clean air''
% J. Environ. Economics & Management, vol.5, 81-102, 1978.
%
clear all;

addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled
%addpath ../ex1/grad_check

% Load housing data from file.
data = load('housing.data');
data=data'; % put examples in columns

% Include a row of 1s as an additional intercept feature.
data = [ ones(1,size(data,2)); data ];

% Shuffle examples.
data = data(:, randperm(size(data,2)));

% Split into train and test sets
% The last row of 'data' is the median home price.
train.X = data(1:end-1,1:400);
train.y = data(end,1:400);

test.X = data(1:end-1,401:end);
test.y = data(end,401:end);

m=size(train.X,2);
n=size(train.X,1);

% Initialize the coefficient vector theta to random values.
theta1 = rand(n,1);



% Run the minFunc optimizer with linear_regression.m as the objective.
%
% TODO:  Implement the linear regression objective and gradient computations
% in linear_regression.m
%
tic;
options = struct('MaxIter', 200);
theta1 = minFunc(@linear_regression, theta1, options, train.X, train.y);
fprintf('Optimization took %f seconds.\n', toc);

% Plot predicted prices and actual prices from training set.
actual_prices = train.y;
predicted_prices1 = theta1'*train.X;

% Print out root-mean-squared (RMS) training error.
train_rms1=sqrt(mean((predicted_prices1 - actual_prices).^2));
fprintf('RMS training error: %f\n', train_rms1);

% Print out test RMS error
actual_prices = test.y;
predicted_prices1 = theta1'*test.X;
test_rms1=sqrt(mean((predicted_prices1 - actual_prices).^2));
fprintf('RMS testing error: %f\n', test_rms1);

% Gradient Check  
average_error1 = grad_check(@linear_regression,theta1,200,train.X,train.y);  
fprintf('Average error :%f\n',average_error1);  

% Plot predictions on test data.
plot_prices=true;
if (plot_prices)
  [actual_prices,I] = sort(actual_prices);
  predicted_prices1=predicted_prices1(I);
  figure;
  plot(actual_prices, 'rx');
  hold on;
  plot(predicted_prices1,'bx');
  legend('Actual Price', 'Predicted Price');
  xlabel('House #');
  ylabel('House price ($1000s)');
end


% Run minFunc with linear_regression_vec.m as the objective.
%
% TODO:  Implement linear regression in linear_regression_vec.m
% using MATLAB's vectorization features to speed up your code.
% Compare the running time for your linear_regression.m and
% linear_regression_vec.m implementations.
%
% Uncomment the lines below to run your vectorized code.
%Re-initialize parameters
theta2 = rand(n,1);
tic;
theta2 = minFunc(@linear_regression_vec, theta2, options, train.X, train.y);
fprintf('Optimization took %f seconds.\n', toc);

% Plot predicted prices and actual prices from training set.
actual_prices = train.y;
predicted_prices2 = theta2'*train.X;

% Print out root-mean-squared (RMS) training error.
train_rms2=sqrt(mean((predicted_prices2 - actual_prices).^2));
fprintf('RMS training error: %f\n', train_rms2);

% Print out test RMS error
actual_prices = test.y;
predicted_prices2 = theta2'*test.X;
test_rms2=sqrt(mean((predicted_prices2 - actual_prices).^2));
fprintf('RMS testing error: %f\n', test_rms2);

% Gradient Check  
average_error2 = grad_check(@linear_regression_vec,theta2,200,train.X,train.y);  
fprintf('Average error :%f\n',average_error2);  

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