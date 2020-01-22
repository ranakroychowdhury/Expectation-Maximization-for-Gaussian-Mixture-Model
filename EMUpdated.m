function [meanVec, cov, w] = EMUpdated(data, k)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% https://stackoverflow.com/questions/11269715/matlabs-sigma-must-be-symmetric-and-positive-definite-error-sometimes-not-mak
format shortg;

[d, n] = size(data);
%s = rng;
meanVec = zeros(d, k);
cov = zeros(d, d, k);
w = ones(1, k) * (1 / k);
L = zeros(n, k);
rsum = zeros(n, 1);
csum = zeros(1, k);
P = zeros(n, k);
div = 1.0/(k-1);
incr = 10^(-6); 
globalSum = -Inf;
globalMean = zeros(d, k);
N = zeros(1, 1);
cnt = 0;
iter = 300;
meanCons = 0.0001;
cntMean = 0;

%Initialize meanVec and covariance matrix
for j = 1 : k
    random = rand(d, n) * 2;
    randData = random.*data;
    for i = 1 : d
        meanVec(i, j) = mean(randData(i,:));
    end
end

for j = 1 : k
    random = rand(d, n) * 2;
    randData = random.*data;
    cov(:,:,j) = (randData * randData')/n;
end

%{
for j = 1 : k
    cov(:,:,j) = cov(:,:,j) * cov(:,:,j)';
end
%}
%disp(meanVec);
%disp(cov);
%disp(w);


%Expectation Maximization
bigSum = 0.0;

while(true)
%while(cnt < iter)
    cnt = cnt + 1;
    if(cnt > 1)
        globalSum = bigSum;
        globalMean = meanVec;
    end
    bigSum = 0.0;
    %disp('Iter ')
    %disp(cnt);
    %Computing log likelihood
    for i = 1 : n
        for j = 1 : k
            %disp(cov(:,:,j));
            %disp(size(data(:,i)'));
            %disp(size((meanVec(:,j)')));
            p = mvnpdf(data(:,i)', meanVec(:,j)', cov(:,:,j)) * w(j);
            %disp(p);
            L(i, j) = p;
        end
        %disp(sum(L(i, :)));
        bigSum = bigSum + log10(sum(L(i, :)));
    end
    format shortg
    disp(bigSum);
        
    %E Step
    for i = 1 : n
        for j = 1 : k
            P(i, j) = L(i, j)/sum(L(i, :));
        end
    end
    %disp(P);

    %M Step
    %Computing prior
    for j = 1 : k
        for i = 1 : n
            csum(1, j) = csum(1, j) + P(i, j);
        end
    end
    N = sum(csum);
    w = csum / N;
    %disp(w);
    %disp(csum);
    %disp(N);
       
    %Update meanVec
    for j = 1 : k
        tempmeanVec = zeros(d, 1);
        for i = 1 : n
            tempmeanVec = tempmeanVec + P(i, j) * data(:,i);
        end
        meanVec(:,j) = tempmeanVec/csum(1, j);
    end
    %disp(meanVec);
    
    %Update covariance
    for j = 1 : k
        mat = zeros(d, d);
        for i = 1 : n
            mat = mat + P(i, j) * (data(:,i) - meanVec(:,j)) * (data(:,i) - meanVec(:,j))';
        end
        cov(:,:,j) = mat / csum(1, j);
        cov(:,:,j) = cov(:,:,j) + 0.001 * eye(d);
    end
    format shortg;
    %disp(cov);
    
    
    %Check for loopbreaking condition
    if(bigSum - globalSum < incr)
        for j = 1 : k
            for dim = 1 : d
                if(abs(meanVec(dim, j) - globalMean(dim, j)) < meanCons)
                    cntMean = cntMean + 1;
                end
            end
        end
        if(cntMean == d * k)
            break;
        end
    end
    %disp(meanVec);
    
    
    %Reinitialize matrices
    cntMean = 0;
    csum = zeros(1, k);
    L = zeros(n, k);
    P = zeros(n, k);
end
disp(cnt);
%rng(s);
end
