function [ out ] = cutImage( imageMatrix, rowThreshold )
%This will cut the image to its proper size based on previous determination
%imageMatrix = imread(imageMatrix) ;
[r k f] = size(imageMatrix) ;
out = imageMatrix(rowThreshold:r,:,:) ;
end

