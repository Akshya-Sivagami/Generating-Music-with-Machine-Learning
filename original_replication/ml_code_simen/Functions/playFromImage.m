function playFromImage(imageClean, imagePressed)

[A, thresholdRow, maxRow] = takeImage(imageClean) ;
[positionVector, AVector] = classifyPiano(A) ;


cleanPiano = cutImage(imageClean, thresholdRow) ;
pressedPiano = cutImage(imagePressed, thresholdRow) ;

RGBMatrixClean = neutralPiano(cleanPiano, positionVector) ;
RGBMatrixDirty = neutralPiano(pressedPiano, positionVector) ;
pressedVector = keysPressed(RGBMatrixClean, RGBMatrixDirty) ;


table = freqTable(positionVector, AVector) ;
freq = listOfFrequenciesToPlay(pressedVector, table) ;






%Added to prevent errors:
for i = 1:length(freq)
    if freq(i) < 37
        freq(i) = 0 ;
    elseif freq(i) > 2000
            freq(i) = 0 ;
    end
end

        

playFrequencies(freq,1)






end