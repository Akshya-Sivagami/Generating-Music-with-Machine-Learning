%% VANILLA NEURAL NETWORK REPLICATION
% Stanford CS229 2018 - Generating Music with Machine Learning
%
% Original released source contains:
%   - makeNeuralMatrix.m : original 104-feature representation
%   - Neuraltest.m       : generation using a trained network ("DeppNet")
%
% The released source does NOT contain the code that originally trained
% DeppNet. Therefore, this script reconstructs that missing training step
% using the architecture described in the project report:
%
%   104 inputs -> 1024 hidden neurons -> 88 piano-note classes
%
% Dataset adaptation:
% The original Pirates of the Caribbean/Synthesia input is unavailable.
% We use the released 202.mid MIDI file as substitute input data.

clear;
clc;

%% ------------------------------------------------------------
% PATHS
% -------------------------------------------------------------

root = 'D:\Akshya\Generating-Music-with-Machine-Learning\original_replication\ml_code_simen';

resultsDir = ...
    'D:\Akshya\Generating-Music-with-Machine-Learning\results\generated_music';

evaluationDir = ...
    'D:\Akshya\Generating-Music-with-Machine-Learning\results\evaluation';

addpath(genpath(root));

if ~exist(resultsDir,'dir')
    mkdir(resultsDir);
end

if ~exist(evaluationDir,'dir')
    mkdir(evaluationDir);
end

rng(229);

fprintf('\n====================================================\n');
fprintf(' VANILLA NEURAL NETWORK REPLICATION\n');
fprintf('====================================================\n\n');


%% ------------------------------------------------------------
% LOAD RELEASED MIDI
% -------------------------------------------------------------

inputFile = fullfile(root,'ReadMidi','Midi','202.mid');

fprintf('Loading: %s\n',inputFile);

Midi = readmidi(inputFile);

MidiInfoMatrix = midiInfo(Midi,0);

% Original ML code works with the first six MIDI columns.
MidiMatrixOriginal = MidiInfoMatrix(:,1:6);

fprintf('Input MIDI events: %d\n',size(MidiMatrixOriginal,1));


%% ------------------------------------------------------------
% CONVERT MIDI PITCH TO 88-KEY PIANO INDEX
% -------------------------------------------------------------
%
% Standard piano:
% MIDI 21  -> piano key 1
% MIDI 108 -> piano key 88

MidiMatrix = MidiMatrixOriginal;

MidiMatrix(:,3) = MidiMatrix(:,3) - 20;

fprintf('Piano-key range: %d - %d\n', ...
    min(MidiMatrix(:,3)),max(MidiMatrix(:,3)));

if any(MidiMatrix(:,3) < 1 | MidiMatrix(:,3) > 88)
    error('Some MIDI notes fall outside the 88-key piano range.');
end


%% ------------------------------------------------------------
% AUTHORS' ORIGINAL FEATURE CONSTRUCTION
% -------------------------------------------------------------

[NeuralMatrix,NeuralVector] = makeNeuralMatrix(MidiMatrix);

fprintf('\nFeature matrix: %d x %d\n', ...
    size(NeuralMatrix,1),size(NeuralMatrix,2));

fprintf('Target matrix:  %d x %d\n', ...
    size(NeuralVector,1),size(NeuralVector,2));

fprintf('Examples: %d\n',size(NeuralMatrix,2));

validTargets = sum(sum(NeuralVector,1) == 1);

fprintf('Valid one-hot targets: %d / %d\n', ...
    validTargets,size(NeuralVector,2));


%% ------------------------------------------------------------
% RECONSTRUCT MISSING NETWORK TRAINING
% -------------------------------------------------------------
%
% Paper architecture:
%   104 inputs
%   1024 hidden neurons
%   88 output classes
%
% 15% validation
% 15% testing
% Remaining 70% training.

fprintf('\nCreating 104 -> 1024 -> 88 neural network...\n');

DeppNet = patternnet(1024);

DeppNet.divideFcn = 'dividerand';

DeppNet.divideParam.trainRatio = 0.70;
DeppNet.divideParam.valRatio   = 0.15;
DeppNet.divideParam.testRatio  = 0.15;

% Paper reports training over 115 iterations.
DeppNet.trainParam.epochs = 115;

% Classification objective.
DeppNet.performFcn = 'crossentropy';

% Keep MATLAB training GUI visible.
DeppNet.trainParam.showWindow = true;

fprintf('Training network...\n\n');

[DeppNet,tr] = train( ...
    DeppNet, ...
    NeuralMatrix, ...
    NeuralVector);


%% ------------------------------------------------------------
% NETWORK PREDICTIONS
% -------------------------------------------------------------

Y = DeppNet(NeuralMatrix);

[~,predictedClass] = max(Y,[],1);
[~,trueClass]      = max(NeuralVector,[],1);

overallAccuracy = ...
    mean(predictedClass == trueClass) * 100;


%% ------------------------------------------------------------
% TRAIN / VALIDATION / TEST METRICS
% -------------------------------------------------------------

trainInd = tr.trainInd;
valInd   = tr.valInd;
testInd  = tr.testInd;

trainAccuracy = mean( ...
    predictedClass(trainInd) == trueClass(trainInd)) * 100;

valAccuracy = mean( ...
    predictedClass(valInd) == trueClass(valInd)) * 100;

testAccuracy = mean( ...
    predictedClass(testInd) == trueClass(testInd)) * 100;


trainLoss = perform( ...
    DeppNet, ...
    NeuralVector(:,trainInd), ...
    Y(:,trainInd));

valLoss = perform( ...
    DeppNet, ...
    NeuralVector(:,valInd), ...
    Y(:,valInd));

testLoss = perform( ...
    DeppNet, ...
    NeuralVector(:,testInd), ...
    Y(:,testInd));


fprintf('\n====================================================\n');
fprintf(' TRAINING COMPLETE\n');
fprintf('====================================================\n');

fprintf('Training samples:   %d\n',length(trainInd));
fprintf('Validation samples: %d\n',length(valInd));
fprintf('Test samples:       %d\n',length(testInd));

fprintf('\nTraining accuracy:   %.2f%%\n',trainAccuracy);
fprintf('Validation accuracy: %.2f%%\n',valAccuracy);
fprintf('Test accuracy:       %.2f%%\n',testAccuracy);
fprintf('Overall accuracy:    %.2f%%\n',overallAccuracy);

fprintf('\nTraining loss:   %.6f\n',trainLoss);
fprintf('Validation loss: %.6f\n',valLoss);
fprintf('Test loss:       %.6f\n',testLoss);


%% ------------------------------------------------------------
% GENERATE MUSIC
% -------------------------------------------------------------
%
% This follows the behavior of the authors' Neuraltest.m:
% retain the original timing/duration information but replace
% the note with the network's predicted note.
%
% makeNeuralMatrix discards the first 100 events and predicts
% the following event.

GeneratedMidiMatrix = MidiMatrixOriginal;

numPredictions = length(predictedClass);

for i = 1:numPredictions

    % Prediction i corresponds to MIDI event i+101.
    midiRow = i + 101;

    if midiRow <= size(GeneratedMidiMatrix,1)

        % Convert piano-key index back to standard MIDI pitch.
        GeneratedMidiMatrix(midiRow,3) = ...
            predictedClass(i) + 20;

    end
end


%% ------------------------------------------------------------
% WRITE GENERATED MIDI
% -------------------------------------------------------------

GeneratedMidi = matrix2midi(GeneratedMidiMatrix);

outputFile = fullfile( ...
    resultsDir, ...
    'vanilla_nn_generated.mid');

writemidi(GeneratedMidi,outputFile);

fprintf('\nGenerated MIDI saved:\n%s\n',outputFile);


%% ------------------------------------------------------------
% SAVE EVALUATION RESULTS
% -------------------------------------------------------------

evaluationFile = fullfile( ...
    evaluationDir, ...
    'vanilla_nn_evaluation.txt');

fid = fopen(evaluationFile,'w');

fprintf(fid,'VANILLA NEURAL NETWORK REPLICATION\n');
fprintf(fid,'==================================\n\n');

fprintf(fid,'Input dataset: 202.mid\n');
fprintf(fid,'Input MIDI events: %d\n', ...
    size(MidiMatrixOriginal,1));

fprintf(fid,'Feature dimensions: %d\n', ...
    size(NeuralMatrix,1));

fprintf(fid,'Output classes: %d\n', ...
    size(NeuralVector,1));

fprintf(fid,'Usable examples: %d\n\n', ...
    size(NeuralMatrix,2));

fprintf(fid,'Architecture: 104 -> 1024 -> 88\n');
fprintf(fid,'Training/validation/test split: 70/15/15\n');
fprintf(fid,'Maximum epochs: 115\n');
fprintf(fid,'Random seed: 229\n\n');

fprintf(fid,'Training samples: %d\n',length(trainInd));
fprintf(fid,'Validation samples: %d\n',length(valInd));
fprintf(fid,'Test samples: %d\n\n',length(testInd));

fprintf(fid,'Training accuracy: %.4f%%\n',trainAccuracy);
fprintf(fid,'Validation accuracy: %.4f%%\n',valAccuracy);
fprintf(fid,'Test accuracy: %.4f%%\n',testAccuracy);
fprintf(fid,'Overall accuracy: %.4f%%\n\n',overallAccuracy);

fprintf(fid,'Training cross-entropy: %.8f\n',trainLoss);
fprintf(fid,'Validation cross-entropy: %.8f\n',valLoss);
fprintf(fid,'Test cross-entropy: %.8f\n',testLoss);

fprintf(fid,'\nNOTE:\n');
fprintf(fid,['The original trained DeppNet and its training script were ' ...
    'not included in the released source code.\n']);
fprintf(fid,['This run reconstructs the missing training stage using the ' ...
    'architecture described in the report while retaining the ' ...
    'authors'' released feature-construction code.\n']);
fprintf(fid,['The original Pirates/Synthesia dataset was unavailable; ' ...
    'released 202.mid was used as substitute data.\n']);

fclose(fid);


%% ------------------------------------------------------------
% SAVE TRAINED NETWORK
% -------------------------------------------------------------

modelFile = fullfile( ...
    evaluationDir, ...
    'vanilla_nn_model.mat');

save(modelFile,'DeppNet','tr');


%% ------------------------------------------------------------
% FINAL SUMMARY
% -------------------------------------------------------------

fprintf('\n====================================================\n');
fprintf(' VANILLA NN REPLICATION COMPLETE\n');
fprintf('====================================================\n');

fprintf('Generated MIDI exists: %d\n', ...
    exist(outputFile,'file'));

fprintf('Evaluation file exists: %d\n', ...
    exist(evaluationFile,'file'));

fprintf('Model file exists: %d\n', ...
    exist(modelFile,'file'));

fprintf('\nGenerated MIDI:\n%s\n',outputFile);
fprintf('\nEvaluation:\n%s\n',evaluationFile);
fprintf('\nModel:\n%s\n',modelFile);