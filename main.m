%% main function 

%% fine-tune cnn

[net, info, expdir] = finetune_cnn();

%% extract features and train svm

all_nets = dir(fullfile('data','fine_tuned_networks','*.mat'));
nets.pre_trained = load(fullfile('data', 'pre_trained_model.mat')); nets.pre_trained = nets.pre_trained.net; 
data = load(fullfile(expdir, 'imdb-stl.mat'));

%% Get accuracies
% for q = 1:length(all_nets) 
%     nets.fine_tuned = load(fullfile('data','fine_tuned_networks',all_nets(q).name)); nets.fine_tuned = nets.fine_tuned.net;
%     train_svm(nets, data, all_nets(q).name);
% end
% 
% play_sound();

%% visualize
net = load(fullfile('data','fine_tuned_networks','50-120.mat')); net = net.net;
net.layers = net.layers(:, 1:end-1);
forward = vl_simplenn(net, data.images.data);
ft_tsne = tsne(squeeze(forward(12).x(:,:,:,2501:6500))', double(data.images.labels(2501:6500))', 2, 64, 50);
keyboard

%%

net = load(fullfile('data','pre_trained_model.mat')); net = net.net;
net.layers = net.layers(:, 1:end-1);
forward = vl_simplenn(net, data.images.data);
pt_tsne = tsne(squeeze(forward(12).x(:,:,:,2501:6500))', double(data.images.labels(2501:6500))', 2, 64, 50);

%%
plot_data(ft_tsne, data.images.labels(2501:6500)', 'Fine-tuned network');
plot_data(pt_tsne, data.images.labels(2501:6500)', 'Pre-trained network');
