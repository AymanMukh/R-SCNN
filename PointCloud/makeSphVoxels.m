
close all
clear

% addpath('utils/', '/media/SSD/DATA/ayman/papers/mnist/mnist')  % augmentation functions

shape_names = {'airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone',...
        'cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar','keyboard','lamp',...
        'laptop','mantel','monitor' 'night_stand','person','piano','plant','radio','range_hood','sink',...
        'sofa','stairs','stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox'};

filename='ply_data_test0.h5'; 
data = h5read(filename,'/data');
label = h5read(filename,'/label');
filename='ply_data_test1.h5';
data=cat(3,data,h5read(filename,'/data'));
label=[label,h5read(filename,'/label')];

% filename='ply_data_train0.h5'; 
% data = h5read(filename,'/data');
% label = h5read(filename,'/label');
% filename='ply_data_train1.h5';
% data=cat(3,data,h5read(filename,'/data'));
% label=[label,h5read(filename,'/label')];
% filename='ply_data_train2.h5';
% data=cat(3,data,h5read(filename,'/data'));
% label=[label,h5read(filename,'/label')];
% filename='ply_data_train3.h5';
% data=cat(3,data,h5read(filename,'/data'));
% label=[label,h5read(filename,'/label')];
% filename='ply_data_train4.h5';
% data=cat(3,data,h5read(filename,'/data'));
% label=[label,h5read(filename,'/label')]; 
 
% number of rotations for training
rep=0;


%%


for ll=0:rep

    outputname=strcat('data/sc/2k/cn7zgn/voxlnonormall18aug',num2str(ll),'.h5');

 tic   
for j=1:length(label)



    
  
     points=data(:,:,j);
         
% rotation while training
%           points=rotate(points,0,rand*2*pi,0);

            %% add outliers , noise , or missing points
%            points=noise(points,.1); %  dont add noise use rotate only
%                  points=outliers(points,.5,[-1 1]);
%      points=missing_points(points,.9);
%    points=pseduo_outliers1(points,.2,.05);
%         points=cluster_outliers(points,.2,10,.04);

 %% find points normals
ptCloud=pointCloud(points');
normals_c = pcnormals(ptCloud,9)';

 % find point-000 vector
v = points./sum(points.^2,1).^.5;
% angle ptw two vectors
thetas= acos(abs(sum(normals_c.*v)));

%    points=noise(points,.01); %  dont add noise use rotate only

if ~isreal(thetas)
thetas=real(thetas);
disp('warning img data')    
end
% -----------------------------------

% normals_c = pcnormals(ptCloud,21)';
% 
%  % find point-000 vector
% v = points./sum(points.^2,1).^.5;
% % angle ptw two vectors
% thetas2= acos(abs(sum(normals_c.*v)));
% 
% %    points=noise(points,.01); %  dont add noise use rotate only
% 
% if ~isreal(thetas2)
% thetas2=real(thetas2);
% disp('warning img data')    
% end

 %% SH

[theta,rho,z]= cart2sph(points(1,:),points(2,:),points(3,:));
points=[theta;rho;z];
[rr,thetas] =   sphvoxels(points,thetas,64,7);  

r=permute(rr, [3,1,2]);
thetas=permute(thetas, [3,1,2]);
 

voxl(:,:,:,j)=r;
voxlth(:,:,:,j)=thetas;



end

toc
% assert(length(voxl)==length(label))

% voxl = cat(1,voxl,voxlth);


 disp(j)
 
 h5create(outputname,'/data',size(voxl),'Datatype','single');
 h5write(outputname,'/data',voxl);
 h5create(outputname,'/theta',size(voxlth),'Datatype','single');
 h5write(outputname,'/theta',voxlth);
%  h5create(outputname,'/theta2',size(voxlth2),'Datatype','single');
%  h5write(outputname,'/theta2',voxlth2);
 h5create(outputname,'/label',size(label),'Datatype','uint8');
 h5write(outputname,'/label',label);

 h5disp(filename);
 voxl=[];voxlth=[]; 
 
end


function [cnt,sz] = sphvoxels(data,thetas,nn,Coce)
% s=[phi-pi;theta-pi/2; ]';

data=data';
p=data;
% p=[p(:,1)+pi,p(:,2)+pi/2,p(:,3);] ;

ntheta=360/nn;
nphi=180/nn;
xedge = (0:ntheta:360)*pi/180; 
yedge = (0:nphi:180)*pi/180; %linspace(min(data(:,2)),max(data(:,2)),bins);
zedge = linspace(0,1.5,Coce+1); 

 xedge=xedge-pi;
 yedge=yedge-pi/2;


loc = zeros(size(data));
len1 = length(xedge)-1;
len2 = length(yedge)-1;
len3 = length(zedge)-1;

[~,loc(:,1)] = histc(p(:,1),xedge);
[~,loc(:,2)] = histc(p(:,2),yedge);
[~,loc(:,3)] = histc(p(:,3),zedge);
hasdata = all(loc>0,2);
sz(1:3) = [len1 len2 len3];
loc1=loc(hasdata,:);
thetas=thetas(hasdata);
data=data(hasdata,:);
% cnt = accumarray(loc(hasdata,:),1,sz);
%cnt = accumarray(loc(hasdata,:),1,sz,@mean);
sz=zeros(sz);
 cnt1=sz;
 loc1(loc1(:,3)>=len3+1,3)=len3;
for i=1:length(thetas)
 sz(loc1(i,1),loc1(i,2),loc1(i,3))=sz(loc1(i,1),loc1(i,2),loc1(i,3))+thetas(i);
cnt1(loc1(i,1),loc1(i,2),loc1(i,3))=cnt1(loc1(i,1),loc1(i,2),loc1(i,3))+1;
end
 cnt=cnt1;
  sz=sz./cnt;
  sz(isnan(sz))=0;
end

function [dd,thet] = imagecast(data,thetas,nn)
% s=[phi-pi;theta-pi/2; ]';

Coce=1;
data=data';
p=data;

ntheta=360/nn;
nphi=180/nn;
xedge = (0:ntheta:360)*pi/180; 
yedge = (0:nphi:180)*pi/180; %linspace(min(data(:,2)),max(data(:,2)),bins);
zedge = linspace(0,2,Coce+1); 

 xedge=xedge-pi;
 yedge=yedge-pi/2;


loc = zeros(size(data));
len1 = length(xedge)-1;
len2 = length(yedge)-1;
len3 = length(zedge)-1;

[~,loc(:,1)] = histc(p(:,1),xedge);
[~,loc(:,2)] = histc(p(:,2),yedge);
[~,loc(:,3)] = histc(p(:,3),zedge);
hasdata = all(loc>0,2);
sz(1:3) = [len1 len2 len3];
loc1=loc(hasdata,:);
thetas=thetas(hasdata);
data=data(hasdata,:);
% cnt = accumarray(loc(hasdata,:),1,sz);
%cnt = accumarray(loc(hasdata,:),1,sz,@mean);

sz=zeros(sz);
 cnt1=sz; dd=sz;
 loc1(loc1(:,3)>=len3+1,3)=len3;
for i=1:length(thetas)
 sz(loc1(i,1),loc1(i,2),loc1(i,3))=sz(loc1(i,1),loc1(i,2),loc1(i,3))+thetas(i);
cnt1(loc1(i,1),loc1(i,2),loc1(i,3))=cnt1(loc1(i,1),loc1(i,2),loc1(i,3))+1;
dd(loc1(i,1),loc1(i,2),loc1(i,3))=max(dd(loc1(i,1),loc1(i,2),loc1(i,3)),data(i,3));

end
 cnt=cnt1;
  sz=sz./cnt;
  sz(isnan(sz))=0;
  thet=sz;
end


% s=[phi-pi;theta-pi/2; ]';

data=data';
p=data;


xedge = linspace(-1,1,nn+1);
yedge = linspace(-1,1,nn+1); %linspace(min(data(:,2)),max(data(:,2)),bins);
zedge = linspace(-1,1,Coce+1); 





loc = zeros(size(data));
len1 = length(xedge)-1;
len2 = length(yedge)-1;
len3 = length(zedge)-1;

[~,loc(:,1)] = histc(p(:,1),xedge);
[~,loc(:,2)] = histc(p(:,2),yedge);
[~,loc(:,3)] = histc(p(:,3),zedge);
hasdata = all(loc>0,2);
sz(1:3) = [len1 len2 len3];
%  loc=loc(hasdata,:);
%  datas=data(hasdata);
 loc(loc(:,1)>nn,1)=nn;loc(loc(:,2)>nn,2)=nn;loc(loc(:,3)>nn,3)=nn;
 
 cnt = accumarray(loc(hasdata,:),1,sz);
%cnt = accumarray(loc(hasdata,:),1,sz,@mean);
% sz=zeros(sz);
%  cnt1=sz;
%  loc1(loc1(:,3)>=len3+1,3)=len3;
% for i=1:length(datas)
% % sz(loc1(i,1),loc1(i,2),loc1(i,3))=sz(loc1(i,1),loc1(i,2),loc1(i,3))+thetas(i);
% cnt1(loc1(i,1),loc1(i,2),loc1(i,3))=cnt1(loc1(i,1),loc1(i,2),loc1(i,3))+1;
% end
%  cnt=cnt1;
 %  cnt=cnt/max(cnt(:));
% sz=sz./cnt;
% sz(isnan(sz))=0;
end

