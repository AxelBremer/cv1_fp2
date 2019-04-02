function plot_data(data, labels, plot_title)
colours = [[0, 0.4470, 0.7410]; [0.8500, 0.3250, 0.0980]; [0.9290, 0.6940, 0.1250]; [0.4940, 0.1840, 0.5560]; [0.4660, 0.6740, 0.1880]];
figure();
hold on
for i = 1:5
    dat = data(labels==i,:);
    scatter(dat(:,1), dat(:,2), 9, colours(i,:), 'filled');
end
legend('airplanes', 'birds', 'ships', 'horses', 'cars');
title(plot_title);
hold off
end

