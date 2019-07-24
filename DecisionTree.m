clear
close all
clc

%% Load the dataset
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
Xtest = X(idtest,:);

X(:,6) = [];
X(:,6) = [];
Xtest(:,6) = [];
Xtest(:,6) = [];

%%
colNames = {'ThreeBetPF','Three3BetSteal','ThreeBetTotal','FourBetPF','FourBetplusTotal','AttToSteal','AvgAllInEquity','Call3Bet','Call4Betplus','CallSteal','CBetF','CBetT','CBetR','XRTotal','FloatTotal','FlopAF','FlopAFq','FoldtoAnyPFR','FoldtoPF3Bet','FoldtoPFSqueeze','FoldtoSteal','FoldvsBTNOpen','PFR','PFRVPIP','PFAF','PFAFq','Limp','PFSqueeze','ProbeTotal','RiverAF','RiverAFq','CallREff','SawTurnPerc','SawRiverPerc','StdDevBB100','Stole','TotalAF','TotalAFq','TurnAF','TurnAFq','VPIP','WinPerc','WSD','CallRandWSD','WSDns','WSDWRR','WSDWRT','WTSDPerc','WWSF','ThreeBetBBvsSB','CallBBvsSB','FoldBBvsSB','CallAnyPFR','CallFBet','CallFCBet','CallFCBetIP','CallFCBetOOP','CallFXR','CallFDonk','CallLPSteal','CallPF3Bet','RaisePFandCall3Bet','CallPF4Betplus','CallPFSqueeze','CBetFSucc','CBetTSucc','CBetRSucc','CC3BetplusPF','DonkF','DonkT','DonkR','Foldto3Bet','Foldto4Betplus','FoldtoFCBet','FoldtoFXR','FoldtoFDonk','FoldtoLPSteal','FoldtoPF4Betplus','LimpBehind','Open4BetplusPF','PFPosAware','RaiseFCBet','RaiseFDonk','RaiseTCBet','RaiseRCBet','RaiseLimpers'};
finalTable = array2table(X,'VariableNames',colNames);
%'AllInAdjBB','AllInAdjBB100',  6,7

%% Alleno il modello su XL e calcolo l'errore su XV, lo calcolo nk volte.
%% Learning the model on XL and error computation on XV, calculated nk times.
nk = 100;
errTrain = zeros(nk,10);
errVal = zeros(nk,10);
prun = 0;
for pruning=0:9
    prun = prun + 1;
    for k=1:nk
        ilearn = idlearn(randperm(nlearn)); 
        ntrain = round(.8*nlearn); 
        nval = round(.2*nlearn);
        itrain = ilearn(1:ntrain); 
        ival = ilearn(ntrain+1:end); 
        XTrain = X(itrain,:);     YTrain = label(itrain); 
        XVal = X(ival,:);         YVal = label(ival); 
        Model = fitctree(XTrain,YTrain); 
        ModelPruned = prune(Model,'Level',pruning);
        YpredTrain = predict(ModelPruned,XTrain);
        YpredVal = predict(ModelPruned,XVal);
        for i = 1:ntrain
            errTrain(k,prun) = errTrain(k,prun) + (YpredTrain(i) ~= YTrain(i));
        end
        for i = 1:nval
            errVal(k,prun) = errVal(k,prun) + (YpredVal(i) ~= YVal(i));
        end
    end
end

%%
errTrainMedio = (sum(errTrain)/nk)/ntrain;
errValMedio = (sum(errVal)/nk)/nval;
errMin = min(errValMedio);
[~,optimalPruningLevel] = min(errValMedio);

%%
xxx = [9,8,7,6,5,4,3,2,1,0];

%% 
figure, hold on, box on, grid on
%x = 0:9;
plot(xxx,errTrainMedio,'-b','linewidth',3);
plot(xxx,errValMedio,'-r','linewidth',3);
plot(6,errMin,'+r','markersize',10,'linewidth',10)
xlabel('tree growth');
ylabel('errors');
legend({'training errors','validation errors'},'Location','southeast','FontSize',16)
title('\fontsize{16}Accuracy of the tree for different levels of pruning')

%% Final test and plotting the tree with optimal pruning level 
%% Plotting also the complete tree
XLearn = X(ilearn,:);  YLearn = label(ilearn);
finalLearn = array2table(XLearn,'VariableNames',colNames);
finalTest = array2table(Xtest,'VariableNames',colNames);
ModelTotal = fitctree(finalLearn,YLearn);
ModelOpt = prune(ModelTotal,'Level',optimalPruningLevel-1); 
YpredFinal = predict(ModelOpt,Xtest);
view(ModelOpt,'Mode','graph')
view(ModelTotal,'Mode','graph')
Confusion = confusionmat(YpredFinal,YTest);

ERRORI = 0;
for i = 1:ntest
    ERRORI = ERRORI + (YpredFinal(i) ~= YTest(i));
end

ERRORI = ERRORI/ntest;
    