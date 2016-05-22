% Points for yellow masking
paleYellowPts = [203/360*255, 21.1, 64.1; 195/360*255, 19.9, 63.1; 38, 5/360*255, 85.5; 48/360*255, 51.4, 55.7; 52/360*255, 72.8, 91.0; 48/360*255, 49.4, 90.6; 62/360*255, 56.4, 86.3; 34/360*255, 3.4, 79.6; 34/360*255, 17.4, 72.2];
brightYellowPtsHex = ['AEBD65','B4C067','FFFFD4','818A50','727C40','9A6D21','AADC8C','BFC866','A2D66F','B1EF7D','5B6134','5F6836','758647','E1FF9D','F2FFB8','FFFFDF'];
pinkPtsHex = ['FF7DFF','B32B6B','B42C67','FF88C9','E252A8','D24E88','B42E4B','FF74A8','D94E77','FF7DDD','BF4465','711F39','FF5FBB', 'FF90B0']; %'FF90B0'
brightYellowPts = rgb2hsv(hex2rgb(brightYellowPtsHex))*255;
pinkPts = rgb2hsv(hex2rgb(pinkPtsHex))*255;

pyhue = paleYellowPts(:, 1);
pysat = paleYellowPts(:, 2);
pyval = paleYellowPts(:, 3);
byhue = brightYellowPts(:, 1);
bysat = brightYellowPts(:, 2);
byval = brightYellowPts(:, 3);
phue = pinkPts(:, 1);
psat = pinkPts(:, 2);
pval = pinkPts(:, 3);

% scatter3(hue, sat, val);
scatter(pyhue, pysat, 'g');
scatter(byhue, bysat, 'b');
scatter(phue, psat, 'm');
xlim([0 255]);
ylim([0 255]);
zlim([0 255]);
xlabel('Hue');
ylabel('Saturation');
title('Post-it Hue & Saturation samples');
zlabel('Value');
grid on
grid minor

scatter(pyhue, pyval, 'g');
scatter(byhue, byval, 'b');
scatter(phue, pval, 'm');
xlim([0 255]);
ylim([0 255]);
zlim([0 255]);
xlabel('Hue');
ylabel('Value');
title('Post-it Hue & Value samples');
zlabel('Value');
grid on
grid minor

[min(phue)-std(phue) max(phue)+std(phue)]
[min(psat)-std(psat) max(psat)+std(psat)]
[min(pval)-std(pval) max(pval)+std(pval)]