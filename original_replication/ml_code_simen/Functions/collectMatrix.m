function out = collectMatrix( Matrix )

%Here we will attempt to go through the matrix piece by piece.

[r, k] = size(Matrix) ;
CollectMatrix = zeros(3) ;
iterator = 1 ;
for column = 2:k
    for i = 2:(r-1)
        
        if Matrix(i,column) == 0
            continue
        end
        
        if (Matrix(i,column) ~= 0) && (Matrix(i-1,column) ~= Matrix(i,column))
            CollectMatrix(iterator, 1) = i ;
            CollectMatrix(iterator,3) = Matrix(i,column) ;
        end
        
        if (Matrix(i,column) ~= 0) && (Matrix(i+1,column) ~= Matrix(i,column))
            CollectMatrix(iterator,2) = i - CollectMatrix(iterator,1) + 1 ; %Plus one for frame count
            iterator = iterator + 1 ;
        end
    end
end

out = CollectMatrix;











end