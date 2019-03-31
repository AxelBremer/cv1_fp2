%% main function 

%% fine-tune cnn

[net, info, expdir] = finetune_cnn();

%% extract features and train svm

all_nets = dir(fullfile('data','fine_tuned_networks','*.mat'));
nets.pre_trained = load(fullfile('data', 'pre_trained_model.mat')); nets.pre_trained = nets.pre_trained.net; 
data = load(fullfile(expdir, 'imdb-stl.mat'));

%% Get accuracies
for q = 1:length(all_nets) 
    nets.fine_tuned = load(fullfile('data','fine_tuned_networks',all_nets(q).name)); nets.fine_tuned = nets.fine_tuned.net;
    train_svm(nets, data, all_nets(q).name);
end

play_sound();

%% visualize
d = load('stl10_matlab/train.mat');
classes = [1, 2, 3, 7, 9];
tr_X = d.X(ismember(d.y, classes),:);
tsne(tr_X, data.images.labels)