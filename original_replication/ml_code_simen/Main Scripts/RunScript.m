%This is the file that we actually run to make the files. 

StartingFrame = 150 ; %This sets which frame the classifier uses. %30 for Beethoven, 50 for Barbie, 120 Virginia
Determinant = 1 ; % 1: Chord, 2: Notes, 3: Both
threshold = 0.7 ; %This is the white value at at which all should become white
fixingParameter = 3 ; %Toggle this if things do not seem to work

VideoMatrix = getFrequencies('Symphony.mp4', 120, Determinant, threshold, fixingParameter) ; 



MidiMatrixChord = generateMidiMatrix(VideoMatrix, 30, 0) ; %the last value is the transpose

 Midi= matrix2midi(MidiMatrix);
 
 writemidi(Midi, 'Despacito.mid');
fprintf('Complete! \n')

% %For MachineLearning:
% 
% 
[NeuralMatrixDespa, NeuralVectorDespa] = makeNeuralMatrix( MidiMatrix ) ;