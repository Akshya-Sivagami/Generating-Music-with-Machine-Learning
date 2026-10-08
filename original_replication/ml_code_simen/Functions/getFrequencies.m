function out = getFrequencies(video, pianoframeNumber, determinant, threshold, fixingParameter)
%video is the input file name and pianiframeNumber is the frame of the
%video to be used for the clean piano

Video = VideoReader(video); %Getting the videofile to be prosessed
imageClean = read(Video,pianoframeNumber) ; %Getting an image to be used in the analysis
imshow(imageClean) %Howing the image to see if it was a good choice

%%%%%           This section is only to show the progress
%Getting the number of frames:
numFrames = ceil(Video.FrameRate*Video.Duration);
numFramesSixty = ceil(numFrames/60) ;


%Getting data from the initial picture:
[BWPiano, thresholdRow, ~] = takeImage(imageClean, threshold, fixingParameter) ;
[positionVector, AVector] = classifyPiano(BWPiano) ;
cleanPiano = cutImage(imageClean, thresholdRow) ;
imshow(cleanPiano)

RGBMatrixClean = neutralPiano(cleanPiano, positionVector) ;
frequencyMatrix = zeros(10) ;
check = floor(numFrames/20) ;
start = 50 ;
clc
fprintf('Analyzing video... \n')
fprintf('Start                                                   Done\n')


for i = start:numFrames
    
    frame = read(Video,i) ;
    frequencies = getFrequency(RGBMatrixClean, frame, thresholdRow, positionVector, AVector, determinant) ;
    frequencyMatrix(i-start+1,1) = i ;
    if ~isempty(frequencies) 
        for j = 1:length(frequencies)
            frequencyMatrix(i,j+1) = frequencies(j) ;
        end 
    end
    
    

    if mod(i,numFramesSixty) == 0
        fprintf('-')
    end
    
end

fprintf('\nSuccessfully made playable matrix \n')
out = frequencyMatrix ;
        
        








end
