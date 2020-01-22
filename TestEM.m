N = 800;
d = 2;
k = 3;
%filename = 'Iris.csv';
%xlRange = 'B2:E151';
%X = xlsread(filename, xlRange);
%disp(X);
% N rows, d columns
%X = [1 2; 3 4; 5 7];
meanMat = zeros(d, k);
covMat = zeros(d, d, k);
z = zeros(1, N);
X = zeros(N, d);
s = rng;

%plot ellipse
NP = 128;
alpha  = 2*pi/NP*(0:NP);
circle = [cos(alpha);sin(alpha)];
ns = 3;

%Fix mean and cov matrix
meanMat = [-10 0 10; -10, 0, 10];
covMat(:,:,1) = [2 0.5; 0.5, 1]; 
for j = 2 : k
    covMat(:,:,j) = covMat(:,:,1);
end
%disp(meanMat);
%disp(covMat);


%Fix class
for i = 1 : N
    z = randi([1,k],1,N);
end
%disp(z);


%Generate Data
for i = 1 : N
    for j = 1 : k
        if(z(i) == j)
            X(i, :) = mvnrnd(meanMat(:,j)', covMat(:,:,j));
            break;
        end
    end
end    


[meanVec, cov, w] = EMUpdated(X', k);
disp(meanVec);
disp(cov);
disp(w);


figure;
plot(X(:,1), X(:,2), '*');
hold;
plot(meanVec(1,:), meanVec(2,:), '+');
plot(meanMat(1,:), meanMat(2,:), 'o');
for j = 1 : k
    x = meanMat(:,j)';
    P = covMat(:,:,j);
    C = chol(P)'; %Choleski method <-????????????
    ellip = ns*C*circle;
    X = x(1)+ellip(1,:);
    Y = x(2)+ellip(2,:);
    plot(X, Y, '-');
end
for j = 1 : k
    x = meanVec(:,j)';
    P = cov(:,:,j);
    C = chol(P)'; %Choleski method <-????????????
    ellip = ns*C*circle;
    X = x(1)+ellip(1,:);
    Y = x(2)+ellip(2,:);
    plot(X, Y, '-');
end
rng(s);