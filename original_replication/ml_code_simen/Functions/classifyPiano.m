function [ pointVector, AVector ] = classifyPiano( inputPiano )
%In this function we will determine the keys associated with each pixel

%Determining size of piano:
[r, k] = size(inputPiano) ;

%Generating a for loop to classify the keys based on the black keys:

iterator = 1 ;
keyVector = 0 ;

row = ceil(r*0.5) ;

for col = 2:k %Use 2 here to avoid faulty code further down
    if inputPiano(row, col) == 0
        keyVector(iterator) = keyVector(iterator) + 1 ;
    end
    
    if (inputPiano(row, col) == 1) && (inputPiano(row, col-1) == 0)
        iterator = iterator + 1 ;
        keyVector(iterator) = 0 ; %To avoid getting error.
    end
end

%Here we now set the threshold to be half the max value
threshold = min(keyVector) + floor(range(keyVector)/2) ;

%Then we will go through the piano once again:

iterator = 1 ;
distanceVector = [] ;
col = 1 ;
check = 1 ;

while col < k
    while (inputPiano(row, col) == 0) && (col < k)
        check = check + 1;
        col = col + 1 ;
    end
    
    if check > threshold
        distanceVector(iterator) = col ;
        iterator = iterator + 1 ;
        distanceVector(iterator) = 0 ; %To avoid error
    end
    check = 1 ;
    col = col + 1 ;
end

%Now we find the relative distances between the black keys
jumpVector = 1 ;
for i = 1:(length(distanceVector)-2)
    jumpVector(i) = distanceVector(i+1)-distanceVector(i) ;
end
jumpThreshold = min(jumpVector) + floor(range(jumpVector)/2) ;
pointVector = 0 ;
iterator = 1 ;
check = 0 ;
AVector = 0 ;
iterator2 = 1 ;

for i = 1:(length(distanceVector)-2)
    jump = distanceVector(i+1) - distanceVector(i) ;
    if jump < jumpThreshold
        step = floor(jump / 4) ;
        pointVector(iterator) = distanceVector(i) + step ;
        iterator = iterator + 1 ;
        pointVector(iterator) = distanceVector(i) + 3*step ;
        iterator = iterator + 1 ;
        check = check + 1 ;
    else
        step = floor(jump / 6) ;
        pointVector(iterator) = distanceVector(i) + step ;
        iterator = iterator + 1 ;
        pointVector(iterator) = distanceVector(i) + 3*step ;
        iterator = iterator + 1 ;
        pointVector(iterator) = distanceVector(i) + 5*step ;
        iterator = iterator + 1 ;
        check = 0 ;
    end
    
    if check == 2
        %Her har vi med en A å gjøre:
        AVector(iterator2) = pointVector(iterator-2) ;
        iterator2 = iterator2 + 1 ;
    end
end
AllVector = pointVector ;






