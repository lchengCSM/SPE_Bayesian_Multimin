% The MIT License
% 
% Copyright © 2022 Liwei Cheng
%
% Permission is hereby granted, free of charge, to any person obtaining a 
% copy of this software and associated documentation files (the “Software”)
% , to deal in the Software without restriction, including without 
% limitation the rights to use, copy, modify, merge, publish, distribute, 
% sublicense, and/or sell copies of the Software, and to permit persons 
% to whom the Software is furnished to do so, subject to the following 
% conditions:
% 
% The above copyright notice and this permission notice shall be included 
% in all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, 
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
% IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
% CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
% TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
% SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%
% The eacorr routine by Dr. Aslak Grinsted is also licensed under the 
% MIT license. See full license within the eacorr.m file itself.

clear;close all
rng(1)       % random seed number for reproducible results
noise=0.02;  % set % of noise 
% do you have optimization tool box? 1=yes 0=no
optimization_tool=0; % this only affects deterministic inversion

%% cly/qtz/wtr
v=[0.35; 0.45; 0.2]; % true answer of one depth point

%% endpoints of cly/qtz/wtr
% rho
rho=[2.79 2.65 1.0];

% nphi
nphi=[0.35 -0.02 1.0];

% endpoint matrix
G = [rho; nphi; ones(1,3)]; 

%% synthetic well log with noise
RHOB=v(1)*rho(1)+v(2)*rho(2)+v(3)*rho(3);
RHOB=RHOB+RHOB.*noise.*randn; % noise

NPHI=v(1)*nphi(1)+v(2)*nphi(2)+v(3)*nphi(3);
NPHI=NPHI+NPHI.*noise.*randn; % noise

D = [RHOB; NPHI; 1]; % data of one depth point

%% normalize data/endpoint
% the input data are normalized by their standard deviation
% in this one depth case, the data and endpoints are normalized by the data
W=[1/D(1) ; 1/D(2); 1];

G = G.*repmat(W,1,3);   % G normalized
D = D.*repmat(W,1,1);   % D normalized

%% deterministic least sqaure solution
% upper/lower bound
ub=[1;1;1];
lb=[0;0;0];

if optimization_tool
    % constrained by upper and lower bounds
    [M,res]=lsqlin(G,D,[],[],[],[],lb,ub);
else
    % unconstrained 
    M=G\D;
    res=(D-G*M)'*(D-G*M);
end
    
%% map model space using grid search
spacing=0.005;
ii=1;
map=zeros(1471,4);
for i=0.2:spacing:0.6
    for j=0.2:spacing:0.6       
        k=1-(i+j);
        m=[i; j; k];
        E=-0.5*(D-G*m)'*(D-G*m);  % objective function in grid search      
        map(ii,1)=i; map(ii,2)=j; map(ii,3)=k; map(ii,4)=E;
        ii=ii+1;
    end
end

bb=map;
ind=find(map(:,3)<0);
map(ind,:)=[];
bb(ind,4)=NaN;

[X,Y] = meshgrid(0.2:spacing:0.6,0.2:spacing:0.6);
Z=X; C=X; 
n=length((0.2:spacing:0.6)');
for i=1:n
    Z(1:n,i)=bb((n*(i-1)+1):n*i,3);
    C(1:n,i)=bb((n*(i-1)+1):n*i,4);
end

Z(Z(:)<0)=0;

%% plot the map of model space
figure(1);set(1,'color','w')
surf(X,Y,Z,C,'EdgeColor','none');alpha(.7)
colormap(jet)
hold on
a1=scatter3(v(1),v(2),v(3),80,'filled','w','MarkerEdgeColor','k');
a2=scatter3(M(1),M(2),M(3),100,'d','filled','k','MarkerEdgeColor','k');
hold off
xlabel('V_{CLY} (v/v)');ylabel('V_{QTZ} (v/v)');zlabel('V_{WTR} (v/v)')
legend([a1  a2],'True answer','Deterministic solution','Location',...
    'northeast','NumColumns',1,'fontsize',15)
colormap(gca,'jet')
h1=colorbar;ylabel(h1,'Misfit','fontsize',15);
h1.Location = 'northoutside'; %caxis([0 .001])
set(gca,'fontsize',12,'colorscale','log')
view(-230,20);grid on

%% mcmc forward function
N=3;            % number of unknowns
sigma=1e-4;     % a constant determining the precision of MCMC simulations
Nwalkers=100;   % numner of walkers
Nstep=500;      % number of total recorded steps per walker
BurnIn=0.3;     % burn % of the steps before reaching targeted area
StepSize=5;     % 5 is a reasonable step size
ThinChain=1;    % only record every # steps

%% Prior & Likelihood functions
% likelihood function
Likelihood=@(m)-0.5*((D-G*m)'*(D-G*m)./sigma) ;

% prior function (all volume fraction solutions should be <1 & >0
Prior =@(m) m(1)>0 & m(1)<1 & m(2)>0 & m(2)<1 & m(3)>0 & m(3)<1 ;

% mcmc function
[Mchain,logP,rate,initprop]=mcmces_m(Prior,Likelihood,N,Nwalkers,Nstep,StepSize,ThinChain);
logP=-logP;

%% burn-in
crop=ceil(Nstep*BurnIn);
ModelsNoBurn=Mchain(:,:);
ModelsBurn=Mchain(:,:,crop+1:end);
ModelsBurn=ModelsBurn(:,:);

logPBurn=logP(:,:,crop+1:end);
logPNoBurn=logP(:,:);

%% plot all results (no burn-in)
gname={"V_{CLY} (v/v)","V_{QTZ} (v/v)","V_{WTR} (v/v)"};

figure(2);set(2,'color','w')
subplot(N+1,1,1)
semilogy(logPNoBurn,'color','k')
ylabel('Log-Likelihood');title('Trace plot')
set(gca,'fontsize',12);grid on
ax = gca;ax.XRuler.Exponent = 0;

for i=1:N    
    subplot(N+1,1,i+1)
    plot(ModelsNoBurn(i,:))  
    hold on 
    plot([Nstep*Nwalkers*BurnIn Nstep*Nwalkers*BurnIn],[0 1],'--k','linewidth',2)
    hold off
    ylabel(gname(i))
    set(gca,'fontsize',12);grid on
    ax = gca;ax.XRuler.Exponent = 0;
end
xlabel('Total steps')

%% plot prior & posterior probablility functions
figure(3);set(3,'color','w')%,'position',[100 200 1000 700])

% prior
for i=1:3
    subplot(1,6,i)
    histogram(initprop(i,:))
    set(gca,'fontsize',12);grid on
    if i==1
        ylabel('Count')
    end
    title('Prior');xlabel(gname(i));
end

% posterior
MAXCOUNT=800;
for i=1:N
    MEAN=mean(ModelsBurn(i,:));
    STD=std(ModelsBurn(i,:));
    subplot(1,6,i+3)
    fill([MEAN-STD MEAN-STD MEAN+STD MEAN+STD],[0 MAXCOUNT MAXCOUNT 0],...
        'y','EdgeColor','none','FaceAlpha',0.5)
    hold on
    histfit(ModelsBurn(i,:))
    
    
    plot([v(i) v(i)],[0 800],'--k','linewidth',1.2)
    hold off
    xlim([0 1]) ;xlabel(gname(i));set(gca,'fontsize',12);grid on
    %     if i==1
    %        ylabel('Count')
    %     end
    text(0.5,640,sprintf('Mean = %1.2f',mean(ModelsBurn(i,:))),'fontsize',14)
    text(0.5,560,sprintf('STD = %1.2f',std(ModelsBurn(i,:))),'fontsize',14)
    title('Posterior')
end

%% autocorrelation
figure(4);set(4,'color','w')

[Corr,lags,ESS]=eacorr(Mchain);
plot(lags,Corr,lags([1 end]),[0 0],'k','linewidth',2);
grid on
xlabel('Lags')
ylabel('Autocorrelation');
title('Markov Chain Auto-Correlation')
legend('V_{CLY}', 'V_{QTZ}', 'V_{WTR}')
set(gca,'fontsize',15)

%% plot the map of model space plus one mcmc walker 
walkerno=10;         % plot the steps of a specific walker 
walker=squeeze(Mchain(:,walkerno,:));
walkerburn=squeeze(Mchain(:,walkerno,crop+1:end));

figure(5);set(5,'color','w')
subplot(1,2,1)
surf(X,Y,Z,C,'EdgeColor','none');alpha(.7)
colormap(jet)
hold on
a1=scatter3(v(1),v(2),v(3),80,'filled','w','MarkerEdgeColor','k');
a2=scatter3(walker(1,1),walker(2,1),walker(3,1),100,'d','filled','g','MarkerEdgeColor','g');
a3=scatter3(walker(1,:),walker(2,:),walker(3,:),10,'filled','r','MarkerEdgeColor','r');
plot3(walker(1,:),walker(2,:),walker(3,:),'k')
hold off
xlabel('V_{CLY} (v/v)');ylabel('V_{QTZ} (v/v)');zlabel('V_{WTR} (v/v)')
legend([a1 a2 a3],'True answer','MCMC initial start point','MCMC steps of a walker','Location',...
    'northeast','NumColumns',1,'fontsize',15)
colormap(gca,'jet')
h1=colorbar;ylabel(h1,'Misfit','fontsize',15);
h1.Location = 'northoutside'; %caxis([0 .001])
set(gca,'fontsize',12,'colorscale','log')
view(-230,20);grid on

subplot(1,2,2)
surf(X,Y,Z,C,'EdgeColor','none');alpha(.7)
colormap(jet)
hold on
scatter3(v(1),v(2),v(3),80,'filled','w','MarkerEdgeColor','k');
a4=scatter3(walkerburn(1,:),walkerburn(2,:),walkerburn(3,:),10,'filled','r','MarkerEdgeColor','r');
hold off
xlabel('V_{CLY} (v/v)');ylabel('V_{QTZ} (v/v)');zlabel('V_{WTR} (v/v)')
legend(a4,'MCMC posterior steps of a walker','Location',...
    'northeast','NumColumns',1,'fontsize',15)
colormap(gca,'jet')
h1=colorbar;ylabel(h1,'Misfit','fontsize',15);
h1.Location = 'northoutside'; %caxis([0 .001])
set(gca,'fontsize',12,'colorscale','log')
view(-230,20);grid on


