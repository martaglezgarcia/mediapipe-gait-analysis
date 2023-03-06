% Corrección de offset en gráfica
clear
clc

name = 'DSCF7682';
joint = 'Left knee flexion';
data = readtable('DSCF7682_Lknee_kinovea.xlsx');

output_data = remove_outliers(data);

time = output_data(:,1);
time = table2array(time);
time = time';
angle = output_data(:,2);
angle = table2array(angle);
angle = angle';

figure(1)
plot(time, angle)
title(joint, name, 'FontSize', 10);
xlabel('time (ms)');
ylabel('angle (º)');

