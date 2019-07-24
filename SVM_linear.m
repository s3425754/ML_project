clear
close all
clc

load('errTrainGauss.mat');
%% Load the dataset
%VARIABILI = "3Bet PF","3Bet Steal","3Bet Total","4Bet PF","4Bet+ Total","All-In Adj BB","All-In Adj BB/100","Att To Steal","Avg All-In Equity","BB Won","BB/100","bb/Hand","Call 3Bet","Call 4Bet+","Call Steal","CBet F","CBet T","CBet R","XR Total","Float Total","Flop AF","Flop AFq","Fold to Any PFR","Fold to PF 3Bet","Fold to PF Squeeze","Fold to Steal","Fold vs BTN Open","PFR","PFR/VPIP","PF AF","PF AFq","Limp","PF Squeeze","Probe Total","River AF","River AFq","Call R Eff","Saw Turn %","Saw River %","StdDev(BB/100)","Stole","Total AF","Total AFq","Turn AF","Turn AFq","VPIP","Win %","WSD","Call R & WSD","WSD (ns)","WSDWRR","WSDWRT","WTSD %","WWSF","3Bet BB v SB","Call BB v SB","Fold BB v SB","Call Any PFR","Call F Bet","Call F CBet","Call F CBet IP","Call F CBet OOP","Call F XR","Call F Donk","Call LP Steal","Call PF 3Bet","2Bet PF & Call 3Bet","Call PF 4Bet+","Call PF Squeeze","CBet F Suc","CBet T Suc","CBet R Suc","CC 3Bet+ PF","Donk F","Donk T","Donk R","Fold to 3Bet","Fold to 4Bet+","Fold to F CBet","Fold to F XR","Fold to F Donk","Fold to LP Steal","Fold to PF 4Bet+","Limp Behind","Open 4Bet+ PF ","PF Pos. Aware","Raise F CBet","Raise F Donk","Raise T CBet","Raise R CBet","Raise Limpers"

X = csvread('MergeReportLast.dat');
X = X(:,4:end);
[N,D] = size(X);
classes = 3;
c = linspace(-1,1,classes+1);
label = zeros(N,1);
occurrences = zeros(classes,1);
BB100 = X(:,11);

%% Assignment of classes & occurrences evaluation using arbitrary values.
for i = 1:N
    if (BB100(i)<-20)
    label(i) = 1;
        occurrences(1)=occurrences(1)+1;
 
    elseif (BB100(i)>=-20 && BB100(i)<20)
    label(i) = 2;  
    occurrences(2)=occurrences(2)+1;
    
    else
    label(i) = 3;
    occurrences(3)=occurrences(3)+1;
    end
end

%% Normalize the dataset
% I normalize to avoid numeric problems due to optimization
for j = 1:D
    mi = min(X(:,j));
    ma = max(X(:,j));
    di = ma - mi;
    if (di < 1.e-6)
        X(:,j) = 0;
    else
        X(:,j) = 2*(X(:,j)-mi)/di-1;
    end
end

%% Drop variables whit a dipendence on BB100 
X(:,[10,11,12]) = [];
[N,D] = size(X);
id = 1:N;
id = id';
X = [id, X];
    
%% Load the test set
load('XTest13408.mat');
load('YTest13408.mat');
idtest = XTest(:,1);
idlearn = id;
idlearn(idtest) = [];
XTest(:,1) = [];
X(:,1) = [];
nlearn = 256;
ntest = 64;

%% MODEL SELECTION to find the best C
%% Learning step
%% All vs All with linear separator
nC = 30; 
nk = 100; 
CC = logspace(-6,4,nC); 
errVal = zeros(nC,1);
errTrain = zeros(nC,1);
ie = 0; 

for C = CC    
    ie = ie + 1;
    for k = 1:nk 
        ilearn = idlearn(randperm(nlearn)); 
        ntrain = round(.8*nlearn); 
        itrain = ilearn(1:ntrain); 
        ival = ilearn(ntrain+1:end);
        XTrain = X(itrain,:);     YTrain = label(itrain); 
        XVal = X(ival,:);         YVal = label(ival); 
        W = []; 
        B = []; 
        for i = 1:classes
            for j = i+1:classes  
                fm = YTrain == i;
                fp = YTrain == j;
                XP = [XTrain(fm,:); XTrain(fp,:)];
                YP = [-ones(sum(fm),1); ones(sum(fp),1)];
                npp = length(YP);
                H = diag(YP)*(XP*XP')*diag(YP);
                [~,~,alpha,b] = ...
                    SMO2_ab(npp,H,-ones(npp,1),YP,zeros(npp,1),C*ones(npp,1),1e+8,1e-4,zeros(npp,1));
                w = XP'*diag(YP)*alpha;
                W = [W, w]; %#ok<AGROW>
                B = [B, b]; %#ok<AGROW> 
            end
        end
        
        YFtrain = []; 
        im = 0; 
        for i = 1:classes
            for j = i+1:classes 
                im = im + 1;
                tmp = XTrain*W(:,im)+B(im);
                tmp(tmp>0) = j;
                tmp(tmp<=0) = i;
                YFtrain = [YFtrain, tmp]; %#ok<AGROW>
            end
        end
        
        YFval = [];
        im = 0; 
        for i = 1:classes
            for j = i+1:classes 
                im = im + 1;
                tmp2 = XVal*W(:,im)+B(im);
                tmp2(tmp2>0) = j;
                tmp2(tmp2<=0) = i;
                YFval = [YFval, tmp2]; %#ok<AGROW>
            end
        end
        YpredTrain = mode(YFtrain,2);
        YpredVal = mode(YFval,2);
        errTrain(ie) = errTrain(ie) + mean(YpredTrain ~= YTrain)/nk; 
        errVal(ie) = errVal(ie) + mean(YpredVal ~= YVal)/nk; 
    end
    fprintf('C: %.6f, ErrVal: %.4f, ErrTrain: %.4f\n',C,errVal(ie),errTrain(ie));
end
[~, i] = min(errVal); 
C = CC(i);
Cbest = C;
errMin = min(errVal);

%%
figure, hold on, box on, grid on
x = logspace(-6,-5,30);
plot(x,errTrain,'-b','linewidth',3);
plot(x,errVal,'-r','linewidth',3);
%plot(x(16),errMin,'+r','markersize',10,'linewidth',10)
xlabel('C');
ylabel('errors');
legend({'training errors','validation errors'},'Location','southeast','FontSize',16)
title('\fontsize{16}Accuracy of the algorithm')

%%
W = [];
B = [];
XLearn = X(ilearn,:);  YLearn = label(ilearn);
for i = 1:classes
    for j = i+1:classes
        fm = YLearn == i;
        fp = YLearn == j;
        XP = [XLearn(fm,:); XLearn(fp,:)];
        YP = [-ones(sum(fm),1); ones(sum(fp),1)];
        n = length(YP);
        H = diag(YP)*(XP*XP')*diag(YP);
        [~,~,alpha,b] = ...
            SMO2_ab(n,H,-ones(n,1),YP,zeros(n,1),C*ones(n,1),1e+8,1e-4,zeros(n,1));
        w = XP'*diag(YP)*alpha;
        W = [W, w]; %#ok<AGROW>
        B = [B, b]; %#ok<AGROW>
    end
end
 
%% Classification step
YF = []; 
        im = 0; 
        for i = 1:classes
            for j = i+1:classes 
                im = im + 1;
                tmp = XTest*W(:,im)+B(im);
                tmp(tmp>0) = j;
                tmp(tmp<=0) = i;
                YF = [YF, tmp]; %#ok<AGROW>
            end
        end
        Ypred = mode(YF,2);

%%
ERRORI = 0;
for i = 1:ntest
    ERRORI = ERRORI + (Ypred(i) ~= YTest(i));
end

ERRORI = ERRORI/ntest;

%%
Confusion = confusionmat(Ypred,YTest);

