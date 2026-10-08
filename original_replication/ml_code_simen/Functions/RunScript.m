
VideoMatrix = getFrequencies('Barbie.mp4', 100) ;
%This function will take in a video file. The number denotes which frame
%should be used as the clean piano. The output is a matrix with each frame
%in a row with the corresponding frequencies in each column.

MidiMatrix = generateMidiMatrix(VideoMatrix, 30) ;

Midi= matrix2midi(MidiMatrix);

writemidi(Midi, 'Barbie.mid');
fprintf('Complete! \n')

%For MachineLearning:

[NeuralMatrixBarbie, NeuralVectorBarbie] = makeNeuralMatrix( MidiMatrix ) ;