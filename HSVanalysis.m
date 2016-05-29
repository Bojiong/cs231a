% Points for yellow masking
paleYellowPtsHex = [ 'C0A989', '998363', 'B2A188', 'BAAE9D', 'D1CDC8', 'DAD6CF', 'CEB896', 'FEE0A4', 'B09C76' ];
brightYellowPtsHex = ['AEBD65','B4C067','FFFFD4','818A50','727C40','9A6D21','AADC8C','BFC866','A2D66F','B1EF7D','5B6134','5F6836','758647','E1FF9D','F2FFB8','FFFFDF'];
pinkPtsHex = ['FF7DFF','B32B6B','B42C67','FF88C9','E252A8','D24E88','B42E4B','FF74A8','D94E77','FF7DDD','BF4465','711F39','FF5FBB', 'FF90B0']; %Bright direct daylight: 'FFB3F6', 'FFB0F1'
brightYellowPts = rgb2hsv(hex2rgb(brightYellowPtsHex))*255;
pinkPts = rgb2hsv(hex2rgb(pinkPtsHex))*255;
paleYellowPts = rgb2hsv(hex2rgb(paleYellowPtsHex))*255;

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
%scatter(byhue, bysat, 'b');
%scatter(phue, psat, 'm');
xlim([0 255]);
ylim([0 255]);
zlim([0 255]);
xlabel('Hue');
ylabel('Saturation');
title('Post-it Hue & Saturation samples');
zlabel('Value');
grid on
grid minor

%scatter(pyhue, pyval, 'g');
%scatter(byhue, byval, 'b');
%scatter(phue, pval, 'm');
%xlim([0 255]);
%ylim([0 255]);
%zlim([0 255]);
%xlabel('Hue');
%ylabel('Value');
%title('Post-it Hue & Value samples');
%zlabel('Value');
%grid on
%grid minor

%[min(phue)-std(phue) max(phue)+std(phue)]
%[min(psat)-std(psat) max(psat)+std(psat)]
%[min(pval)-std(pval) max(pval)+std(pval)]

[min(pyhue)-std(pyhue) max(pyhue)+std(pyhue)]
[min(pysat)-std(pysat) max(pysat)+std(pysat)]
[min(pyval)-std(pyval) max(pyval)+std(pyval)]