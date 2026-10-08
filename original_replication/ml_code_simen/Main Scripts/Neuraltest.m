%In this script we will try to visualise what the neural network is doing:

%Generate Midi:
MidiMatrix = generateMidiMatrix(VideoMatrix, 30, 0) ;
MidiMatrix(1:101,:) = [] ; %Cut the first hundred:
[r,k] = size(MidiMatrix) ;
MidiMatrix(r,:) = [] ;
[r,k] = size(MidiMatrix) ;

for i = 1:1630
    %index = NeuralMatrixBarbie(1,i+1) ;
    [val, index] = max(DeppNet(NeuralMatrixDeppLearning(:,i))) ;
    %[val, index] = max(NeuralVectorBarbie(:,i)) ;
    MidiMatrix(i,3) = index ;
    clc
    disp(i)
    
end


Midi= matrix2midi(MidiMatrix);
writemidi(Midi, 'DeppLearningToTest.mid');
fprintf('Complete! \n')