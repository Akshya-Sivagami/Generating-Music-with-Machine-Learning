function [ NewMatrix, thresholdRow, maxRow ] = takeImage( imageFile, threshold, fixingParameter )
%This function will accept an image and will output a dsired matrix:
%This is meant for a representable image so that all other images can be
%cropped.


%Converting the file to a matrix:

%LogicalMatrix = im2bw(rgb2gray(imread(imageFile)),threshold) ;
LogicalMatrix = im2bw(rgb2gray(imageFile),threshold) ;
imshow(LogicalMatrix)


%First it will find the start of the piano:
[r, k] = size(LogicalMatrix) ;
maxRow = r ;

%setting the parameter to zero
parameter = 0 ;
row = floor(r*0.7) ;
col = floor(k*0.9) - fixingParameter ;

while (parameter == 0) 
    parameter = LogicalMatrix(row,col) ;
    if row == r
        row = floor(r/2) ;
        col = col + 5 ;
    else
        row = row+1 ;
    end
end
thresholdRow = row ;

NewMatrix = LogicalMatrix(row:r,:) ;

