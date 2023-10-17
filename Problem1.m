%   Intelligent Systems Homework 2
%   Problem 1: Binary Threshold Neuron
%   Benjamin Middleton

clear all;
close all;
clc;

% read data
disp('Reading Data...');
dataFileID = fopen('HW2_datafiles/MNISTnumImages5000_balanced.txt');
dataFormatSpec = '%f';
labelFileID = fopen('HW2_datafiles/MNISTnumLabels5000_balanced.txt');
labelFormatSpec = '%d';
raw_data = fscanf(dataFileID, dataFormatSpec);
labels = fscanf(labelFileID, labelFormatSpec);

% reshape the X by 1 column vector into a 5000 by 784 matrix
disp('Formatting Data...');
i = 1;
j = 1;
data = zeros(5000, 784);
while i < length(raw_data)
    row = raw_data(i:i+783)';
    data(j, :) = row;
    i = i + 784;
    j = j + 1;
end
clear raw_data dataFileID labelFileID dataFormatSpec labelFormatSpec row i j;

% separate out images for each digit
disp('Separating and organizing data...');
digits = {0 1 2 3 4 5 6 7 8 9};
data_dict = containers.Map(digits, {[] [] [] [] [] [] [] [] [] []});
for i = 1:5000
    if labels(i) == 0
        data_dict(0) = [data_dict(0); data(i, :)];
    elseif labels(i) == 1
        data_dict(1) = [data_dict(1); data(i, :)];
    elseif labels(i) == 2
        data_dict(2) = [data_dict(2); data(i, :)];
    elseif labels(i) == 3
        data_dict(3) = [data_dict(3); data(i, :)];
    elseif labels(i) == 4
        data_dict(4) = [data_dict(4); data(i, :)];
    elseif labels(i) == 5
        data_dict(5) = [data_dict(5); data(i, :)];
    elseif labels(i) == 6
        data_dict(6) = [data_dict(6); data(i, :)];
    elseif labels(i) == 7
        data_dict(7) = [data_dict(7); data(i, :)];
    elseif labels(i) == 8
        data_dict(8) = [data_dict(8); data(i, :)];
    elseif labels(i) == 9
        data_dict(9) = [data_dict(9); data(i, :)];
    end
end
data = data_dict;
clear i data_dict digits labels

% create 0/1 training and test sets
perm = randperm(500);
training_rows = perm(1:400);
test_rows = perm(401:500);
zero_data = data(0);
one_data = data(1);

training_01 = [zero_data(training_rows, :); one_data(training_rows, :)];
labels_training_01 = [zeros(400, 1); ones(400, 1)];
perm = randperm(800);
training_01 = training_01(perm, :);
labels_training_01 = labels_training_01(perm, :);

test_01 = [zero_data(test_rows, :); one_data(test_rows, :)];
labels_test_01 = [zeros(100, 1); ones(100, 1)];

% create challenge set
challenge_set = [];
labels_challenge = [];
for i=2:9
    digit_data = data(i);
    challenge_set = [challenge_set; digit_data(test_rows, :)];
    labels_challenge = [labels_challenge; i*ones(100, 1)];
end
clear i test_rows perm training_rows zero_data one_data digit_data

% train neuron
disp('Training Binary Threshold Neuron...')
initial_w = rand(784, 1)*0.5;
w = initial_w;
eta = 0.01;
theta = -1; % need to pass some arbitrary theta during training
for epoch=1:20
    for i=1:height(training_01)
        z = labels_training_01(i, :);
        x = training_01(i, :)';
        y = binary_threshold_neuron(w, x, theta, z);
        w_next = zeros(height(w), 1);
        for j=1:height(w)
            w_next(j) = w(j) + eta*y*(x(j)-w(j));
        end
        w = w_next;
    end
end
clear w_next i j x y z theta epoch eta

% test for each theta
disp('Testing effectiveness for thetas 0-40...')
z = -1; % turn off teaching input
tp = zeros(41, 1);
fp = zeros(41, 1);
tn = zeros(41, 1);
fn = zeros(41, 1);
precision = zeros(41, 1);
recall = zeros(41, 1);
f1 = zeros(41, 1);
for theta=0:40
    for i=1:height(test_01)
        x = test_01(i, :)';
        y = binary_threshold_neuron(w, x, theta, z);
        if y == 1 && labels_test_01(i) == 1
            tp(theta+1) = tp(theta+1) + 1;
        elseif y == 0 && labels_test_01(i) == 0
            tn(theta+1) = tn(theta+1) + 1;
        elseif y == 1 && labels_test_01(i) == 0
            fp(theta+1) = fp(theta+1) + 1;
        elseif y == 0 && labels_test_01(i) == 1
            fn(theta+1) = fn(theta+1) + 1;
        end
        precision(theta+1) = tp(theta+1)/(tp(theta+1)+fp(theta+1));
        recall(theta+1) = tp(theta+1)/(tp(theta+1)+fn(theta+1));
        f1(theta+1) = 2 * (precision(theta+1)*recall(theta+1)) / (precision(theta+1)+recall(theta+1));
    end
end
clear x y z theta i

% plot performance metrics and roc curves
disp('Plotting...')
theta = 0:40;
tp_rate = tp/height(test_01)*2;
fp_rate = fp/height(test_01)*2;

% performance metrics
f = figure;
f.Position = [100 100 650 600];
plot(theta, precision);
hold on;
plot(theta, recall);
hold on;
plot(theta, f1);
hold off;
title('Performance Measures vs. Theta');
xlabel('Theta');
ylabel('Performance Measure');
legend('Precision', 'Recall', 'F1');

% roc curve
f = figure;
f.Position = [100 100 650 600];
plot(fp_rate, tp_rate);
title('ROC Curve');
xlabel('False Positive Rate');
ylabel('True Positive Rate');

% plot weights before and after training
initial_w_square = reshape(initial_w, 28, 28);
w_square = reshape(w, 28, 28);
f = figure;
f.Position = [100 100 550 500];
heatmap(initial_w_square);
title('Initial (Random) weight heatmap');
xlabel('x');
ylabel('y');

f = figure;
f.Position = [650 100 550 500];
heatmap(w_square);
title('Final weight heatmap');
xlabel('x');
ylabel('y');

clear f w_square initial_w_square

% present challenge set
disp('Presenting challenge set...')
[~, i] = max(f1);
theta_optimal = theta(i);
z = -1;
challenge_result = zeros(2, 8);
for i=1:height(challenge_set)
    x = challenge_set(i, :)';
    y = binary_threshold_neuron(w, x, theta_optimal, z);
    col = floor((i-1)/100)+1;
    if y == 0
        challenge_result(1, col) = challenge_result(1, col) + 1;
    else
        challenge_result(2, col) = challenge_result(2, col) + 1;
    end
end
disp(challenge_result);
clear i x y z col

% binary threshold neuron implementation
% param theta: threshold. can pass anything during training
% param z: teaching input. should be -1 after the training phase
function y = binary_threshold_neuron(w, x, theta, z)
    if z ~= -1
        y = z;
        return;
    end
    s = dot(w, x);
    if s > theta
        y = 1;
        return;
    else
        y = 0;
        return;
    end
end

