function out =  keysPressed(RGBMatrixClean, RGBMatrixDirty, determinant)
%This function will look at what keys were pressed for a given image:
%We will give a value to compare to, and thus we need the percentage
%difference:
threshold = 50 ;
[r ~] = size(RGBMatrixClean) ;



pressedVector = 0 ;

for i = 1:r
    
    for j = 1:3
        %Will here find the percentage difference
        if abs(RGBMatrixClean(i,j) - RGBMatrixDirty(i,j)) > threshold
            if determinant == 1 % Chord
                
                if (RGBMatrixDirty(i,3)/RGBMatrixDirty(i,2) > 1)
                    pressedVector(i) = 1 ;
                end
            elseif determinant == 2 % Note
                if ~(RGBMatrixDirty(i,3)/RGBMatrixDirty(i,2) > 1)
                    pressedVector(i) = 1 ;
                end
            elseif determinant == 3 %Both
                pressedVector(i) = 1 ;
            end
            
            
            
        end
    end    
end

%Make the last element to be a zer0:
%pressedVector(length(pressedVector)) = 0 ;

out = pressedVector ;

end