v = VideoReader('Faded.mp4');
v1 = read(v,50) ;
v2 = read(v,190) ;


[A, thresholdRow, maxRow] = takeImage(v1) ;

[positionVector, AVector] = classifyPiano(A) ;


cleanPiano = cutImage(v1, maxRow, thresholdRow) ;

pressedPiano = cutImage(v2, maxRow, thresholdRow) ;


RGBMatrixClean = neutralPiano(cleanPiano, positionVector) ;
RGBMatrixDirty = neutralPiano(pressedPiano, positionVector) ;
pressedVector = keysPressed(RGBMatrixClean, RGBMatrixDirty) ;
sum(pressedVector)

table = freqTable(positionVector, AVector) ;
freq = listOfFrequenciesToPlay(pressedVector, table) ;


playFrequencies(freq,1)
