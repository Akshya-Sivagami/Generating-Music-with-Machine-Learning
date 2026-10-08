function out =  getFrequency(RGBMatrixClean, frame, thresholdRow, positionVector, AVector, determinant)

pressedPiano = cutImage(frame, thresholdRow) ;

RGBMatrixDirty = neutralPiano(pressedPiano, positionVector) ;
pressedVector = keysPressed(RGBMatrixClean, RGBMatrixDirty, determinant) ;
%Pressedvector returns vector of pressed keys

table = freqTable(positionVector, AVector) ;
out = listOfFrequenciesToPlay(pressedVector, table) ;

end

