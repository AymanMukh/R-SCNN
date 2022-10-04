


close all
clear

d = load('mnist.mat');

Xtrain = d.trainX;
Xtrain = reshape(Xtrain, [],28,28);
Ytrain = d.trainY;

Xtest = d.testX;
Xtest = reshape(Xtest, [],28,28);
Ytest = d.testY;

% image(squeeze(i(3,:,:))');

% s= rng(11112221 );
 
rep=0;

%%

for ll=0:rep

    outputname=strcat('data/mnist/voxl',num2str(ll),'.h5');

 tic   
for j=1:length(Ytrain)


if   1==1 % any(label(j)==shape) %  label1==shape 
    
     
%   disp(j)

     points=squeeze(Xtrain(j,:,:));

%         points=rotate(points,0,rand*2*pi,0);  

%  image(points);
% imshow(mat2gray(points'))


            %% add outliers , noise , or missing points
%  n= 100;
%  outliers=randi([1,784],n,1);
%  points(outliers)=abs(randn(n,1))*250;

 points= reshape(points,1,28,28);% points= reshape(points,28,28)';
 

 %% SH

voxl(:,:,:,j)=points;


end
end

toc
assert(length(voxl)==length(Ytrain))

% voxl = cat(1,voxl,voxlth);

 disp(j)
 
%  h5create(outputname,'/data',size(voxl),'Datatype','single');
%  h5write(outputname,'/data',voxl);
%  h5create(outputname,'/label',size(Ytrain),'Datatype','uint8');
%  h5write(outputname,'/label',Ytrain);
% 
%  h5disp(outputname);
 voxl=[];voxlth=[]; 
end


clear j voxl points
 outputname=strcat('data/mnist/voxlt','.h5');
for j=1:length(Ytest)
if   1==1 % any(label(j)==shape) %  label1==shape 
    
     
%  disp(j)

     points=squeeze(Xtest(j,:,:));

%         points=rotate(points,0,rand*2*pi,0);  

%  image(points);
%  imshow(mat2gray(points'))


            %% add outliers , noise , or missing points
%  n= 100;
%  outliers=randi([1,784],n,1);
%  points(outliers)=abs(randn(n,1))*250;


 points= reshape(points,1,28,28);% points= reshape(points,28,28)';



 %% SH

voxl(:,:,:,j)=points;


end
end

toc
assert(length(voxl)==length(Ytest))
 disp(j)
%  h5create(outputname,'/data',size(voxl),'Datatype','single');
%  h5write(outputname,'/data',voxl);
%  h5create(outputname,'/label',size(Ytest),'Datatype','uint8');
%  h5write(outputname,'/label',Ytest);
%  h5disp(outputname);

 
clear j voxl points
 n= 100;
 outputname=strcat('data/mnist/voxltO100','.h5');

 
for j=1:length(Ytest)
if   1==1 % any(label(j)==shape) %  label1==shape 
    
     
%  disp(j)

     points=squeeze(Xtest(j,:,:));

%         points=rotate(points,0,rand*2*pi,0);  




            %% add outliers , noise , or missing points

 outliers=randi([1,784],n,1);
 points(outliers)=abs(randn(n,1))*250;
 
 points= reshape(points,1,28,28);% points= reshape(points,28,28)';

%  image(points);

 %% SH

voxl(:,:,:,j)=points;


end
end

toc
assert(length(voxl)==length(Ytest))
 disp(j)
%  h5create(outputname,'/data',size(voxl),'Datatype','single');
%  h5write(outputname,'/data',voxl);
%  h5create(outputname,'/label',size(Ytest),'Datatype','uint8');
%  h5write(outputname,'/label',Ytest);
%  h5disp(outputname);

 clear j voxl points
 n= 300;
 outputname=strcat('data/mnist/voxltO300','.h5');
for j=1:length(Ytest)
if   1==1 % any(label(j)==shape) %  label1==shape 
    
     
%  disp(j)

     points=squeeze(Xtest(j,:,:));

%         points=rotate(points,0,rand*2*pi,0);  




            %% add outliers , noise , or missing points

 outliers=randi([1,784],n,1);
 points(outliers)=abs(randn(n,1))*250;
 % imshow(mat2gray(points'))
 %  image(points);

 points= reshape(points,1,28,28);% points= reshape(points,28,28)';


 %% SH

voxl(:,:,:,j)=points;


end
end

toc
assert(length(voxl)==length(Ytest))
 disp(j)
%  h5create(outputname,'/data',size(voxl),'Datatype','single');
%  h5write(outputname,'/data',voxl);
%  h5create(outputname,'/label',size(Ytest),'Datatype','uint8');
%  h5write(outputname,'/label',Ytest);
%  h5disp(outputname);
 
 
 




 layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
 
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
 
 options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');

Xtrain=permute(Xtrain,[2,3,1]);
Xtrain=reshape(Xtrain,28,28,1,[]);

Xtest1=permute(Xtest,[2,3,1]);
Xtest1=reshape(Xtest1,28,28,1,[]);

net = trainNetwork(Xtrain, categorical(Ytrain'),layers,options);

YPred = classify(net,Xtest1);
accuracy = sum(YPred == categorical(Ytest'))/numel(categorical(Ytest'))

Xtest1=permute(voxl,[2,3,1,4]);


 
 
 