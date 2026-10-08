function out = neutralPiano( untouchedPianoImage, positionVector)
% This function will return a set of RGB values for the untouched piano.
distanceDown = 40 ; %This sets the number of rows down from the top

RGBMatrix = zeros(3,3) ;

for i = 1:length(positionVector)
    for j = 1:3
        RGBMatrix(i,j) = untouchedPianoImage(distanceDown, positionVector(i), j) ;
        %Here we will fill out the RGB Values for each key:
    end
end

out = RGBMatrix ;