function out = listOfFrequenciesToPlay(pressedVector, freqTable)
%This function outputs the frequencies that we wish to play based on the
%pressedVector. Output will be a vector in hertz.


frequencies = 0 ;
iterator = 1 ;

for i = 1:length(pressedVector)
    if pressedVector(i) == 1
        frequencies(iterator) = freqTable(i) ;
        iterator = iterator + 1 ;
    end
end


out = frequencies ;
end