clear all;
% Statistical Problem: Baseball Data

% Batting average against
AVG = [209 210 190 211 243 202 211 211 229 241 250 214 219 221 249 250 ...
    218 232 252 230 264 228 233 251 230 225 256 237 239 236]';
AVG = AVG/1000;

% Walks plus hits per inning pitched
WHIP = [106 100 90 102 104 98 116 97 105 127 121 116 104 106 122 118 ...
    117 112 122 111 121 121 101 115 134 112 116 118 117 120]';
WHIP = WHIP/100;

set(0,'DefaultFigurePaperSize',[5 4]);
set(0,'DefaultFigurePaperPosition',[0 0 5 4]);

figure(1);
plot(AVG,WHIP,'ko');
xlim([.18,.28]);
ylim([.8,1.4]);                  
xlabel('Batting Average Against');
ylabel('Walks Plus Hits per Inning Pitched');
title('Baseball Data');
saveTightFigure(figure(1),'scatter1.pdf');

A = [ones(size(AVG)) AVG]; 
beta = (A'*A)\(A'*WHIP)

hold on;
[~,i] = min(AVG);
[~,j] = max(AVG);
plot(AVG([i j]),A(ismember(AVG,AVG([i j])),:)*beta,'b-');
title('Simple Linear Regression Fit');
saveas(figure(1),'scatter2.pdf');

resid = WHIP - A*beta; 
figure(2)
plot(AVG,resid,'ko');
hold on; 
plot(AVG([i j]),[0 0],'b-') 
xlim([.95*min(AVG) 1.05*max(AVG)])
xlabel('Batting Average Against')
ylabel('Residual')
title('Residual Plot')
saveTightFigure(figure(2),'resid.pdf'); 
figure(3)
qqplot(resid)
saveTightFigure(figure(3),'qqplot.pdf');
