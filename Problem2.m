%   Intelligent Systems Homework 2
%   Problem 2: Perceptron
%   Benjamin Middleton

% Since this problem is very similar to problem 1, much of this code is the
% same as Problem1.m in this same submission.

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

% calculate performance before training
disp('Calculating Performance...');
initial_w = rand(785, 1)*0.5; % initialize weights randomly
w = initial_w;
tp = zeros(2, 1);
fp = zeros(2, 1);
tn = zeros(2, 1);
fn = zeros(2, 1);
precision = zeros(2, 1);
recall = zeros(2, 1);
f1 = zeros(2, 1);
for i=1:height(test_01)
    x = [1; test_01(i, :)'];
    y = perceptron(w, x);
    if y == 1 && labels_test_01(i) == 1
        tp(1) = tp(1) + 1;
    elseif y == 0 && labels_test_01(i) == 0
        tn(1) = tn(1) + 1;
    elseif y == 1 && labels_test_01(i) == 0
        fp(1) = fp(1) + 1;
    elseif y == 0 && labels_test_01(i) == 1
        fn(1) = fn(1) + 1;
    end
    precision(1) = tp(1)/(tp(1)+fp(1));
    recall(1) = tp(1)/(tp(1)+fn(1));
    f1(1) = 2 * (precision(1)*recall(1)) / (precision(1)+recall(1));
end
clear x y i

% train perceptron
disp('Training Perceptron...')
eta = 0.01;
epochs = 10;
training_error_fraction = zeros(epochs, 1);
test_error_fraction = zeros(epochs, 1);
for epoch=1:epochs
    for i=1:height(training_01)
        x = [1; training_01(i, :)'];
        y = perceptron(w, x);
        w_next = zeros(height(w), 1);
        for j=1:height(w)
            w_next(j) = w(j) + eta*(labels_training_01(i)-y)*x(j);
        end
        w = w_next;
    end
    % calculate error fraction AFTER every epoch
    training_error_count = 0;
    test_error_count = 0;
    for i=1:height(training_01)
        x = [1; training_01(i, :)'];
        y = perceptron(w, x);
        if y ~= labels_training_01(i)
            training_error_count = training_error_count + 1;
        end
    end
    for i=1:height(test_01)
        x = [1; test_01(i, :)'];
        y = perceptron(w, x);
        if y ~= labels_test_01(i)
            test_error_count = test_error_count + 1;
        end
    end
    training_error_fraction(epoch) = training_error_count/height(training_01);
    test_error_fraction(epoch) = test_error_count/height(test_01);
end
clear w_next i j x y epoch eta training_error_count test_error_count

% calculate performance after training
disp('Calculating Performance...');
for i=1:height(test_01)
    x = [1; test_01(i, :)'];
    y = perceptron(w, x);
    if y == 1 && labels_test_01(i) == 1
        tp(2) = tp(2) + 1;
    elseif y == 0 && labels_test_01(i) == 0
        tn(2) = tn(2) + 1;
    elseif y == 1 && labels_test_01(i) == 0
        fp(2) = fp(2) + 1;
    elseif y == 0 && labels_test_01(i) == 1
        fn(2) = fn(2) + 1;
    end
    precision(2) = tp(2)/(tp(2)+fp(2));
    recall(2) = tp(2)/(tp(2)+fn(2));
    f1(2) = 2 * (precision(2)*recall(2)) / (precision(2)+recall(2));
end
clear x y i

% plots
disp('Plotting...')

% training and test error
f = figure;
f.Position = [100 100 650 600];
epochs = 1:epochs;
plot(epochs, training_error_fraction);
hold on;
plot(epochs, test_error_fraction);
hold off;
title('Training and Test Error vs. Epoch');
xlim([1 width(epochs)]);
ylim([0 0.01]);
xlabel('Epoch Number');
ylabel('Error Fraction');
legend('Training Error', 'Test Error');

% performance metrics
f = figure;
f.Position = [100 100 650 600];
metric_names = categorical({'Precision','Recall','F1-Score'});
metrics = [precision(1) precision(2); recall(1) recall(2); f1(1) f1(2)];
bar(metric_names, metrics);
title('Performance Measures Before and After Training');
xlabel('Metric');
ylabel('Measure');
legend('Before Training', 'After Training', 'Location', 'southeast');
clear metric_names metrics

% plot weights before and after training
initial_w_square = reshape(initial_w(2:end), 28, 28);
w_square = reshape(w(2:end), 28, 28);
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
challenge_result = zeros(2, 8);
for i=1:height(challenge_set)
    x = [1; challenge_set(i, :)'];
    y = perceptron(w, x);
    col = floor((i-1)/100)+1;
    if y == 0
        challenge_result(1, col) = challenge_result(1, col) + 1;
    else
        challenge_result(2, col) = challenge_result(2, col) + 1;
    end
end
disp(challenge_result);
clear i x y col

% perceptron implementation
% x should be [1; x] so that x_0 is the bias 
function y = perceptron(w, x)
    s = dot(w, x);
    if s > 0
        y = 1;
        return;
    else
        y = 0;
        return;
    end
end

