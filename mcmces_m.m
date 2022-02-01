function [Mchain,logP,rate,initprop]=mcmces_m(Prior,Likelihood,N,Nwalkers,Nkeep,StepSize,ThinChain)
% 
% INPUTS:
%  Prior: prior information (all volume fraction solutions should be <1 & >0
%  Likelihood: the likelihood function
%  N: number of unknowns
%  Nwalkers: numner of walkers (chains, >10)
%  Nstep: number of total recorded steps per walker
%  StepSize: the size of the proposed steps (low maintenance)
%  ThinChain: only record every # steps
%
% OUTPUTS:
%  Mchain: A N*Nwalkers*Nstep matrix Markvo chains (without burn-in)
%  logP: A N*Nwalkers*Nstep matrix of log probabilities for each model in 
%    the models
%  rate: acceptance rate
%  initwalker: initial proposals
%
%
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
% Modified from Aslak Grinsted
% https://www.mathworks.com/matlabcentral/fileexchange/49820-ensemble-mcmc-sampler


% initial walker
initprop=rand(N,Nwalkers);   

Mchain=nan(N,Nwalkers,Nkeep); 
Mchain(:,:,1)=initprop;

%calculate logP state initial pos of walkers
% tic
logP=zeros(1,Nwalkers,Nkeep);
for wix=1:Nwalkers
    logP(1,wix,1)=Likelihood(initprop(:,wix));    
end

curm=Mchain(:,:,1);
curlogP=logP(:,:,1);
% acceptance
reject=zeros(Nwalkers,1); accept=zeros(Nwalkers,1); total=zeros(Nwalkers,1);

%%
for row=1:Nkeep
%     row
    for jj=1:ThinChain
        
        rix=mod((1:Nwalkers)+floor(rand*(Nwalkers-1)),Nwalkers)+1;
        zz=((StepSize - 1)*rand(1,Nwalkers) + 1).^2/StepSize;
        proposedm=curm(:,rix) - bsxfun(@times,(curm(:,rix)-curm),zz);
        logrand=log(rand(2,Nwalkers));
        
        for wix=1:Nwalkers
            
            acceptstep=true;
                                
            if logrand(1,wix) < log(zz(wix))
                             
                if Prior(proposedm(:,wix))
                                        
                    proposedlogP=Likelihood(proposedm(:,wix)); %%
                    
                    if logrand(2,wix) > proposedlogP-curlogP(wix)
                        
                        acceptstep=false;                        
                    end
                else
                    acceptstep=false;
                end
                
            else
                acceptstep=false;
            end
            
            if acceptstep
                curm(:,wix)=proposedm(:,wix);
                curlogP(:,wix)=proposedlogP;
                accept(wix)=accept(wix)+1;
            else
                reject(wix)=reject(wix)+1;
            end
            total(wix)=total(wix)+1;
        end

    end
    Mchain(:,:,row)=curm;
    logP(:,:,row)=curlogP;
end
% toc
rate=mean(accept./total);

