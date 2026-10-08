function out = generateFile( PlayableMatrix )

%Dette er bare rot
Fs = 44100 ;
fps = 30 ;
fpsin = 1/fps ;
samplesPerFrame = floor(Fs/fps) ;
durationFrames = max(PlayableMatrix(:,1)) + max(PlayableMatrix(:,2)) ;
musicVector = zeros(1, floor(durationFrames*samplesPerFrame + 100000)) ;
[r, k] = size(PlayableMatrix) ;
soundVector = [ 1 0.5 0.5 0.5 ] ;
fprintf('\nAnalyzing generated matrix... \n')
fprintf('Start                                                   Done\n')
r = ceil(r/10) ;
rCeil60 = ceil(r/60) ;

for i = 1:r
    duration = PlayableMatrix(i,2) ;
    timeVector = linspace(1/samplesPerFrame, duration, duration*samplesPerFrame) ;
    waveVector = zeros(1,floor(samplesPerFrame*duration)) ;
    for j = 1:length(soundVector)
        waveVector = waveVector + soundVector(j) * cos( fpsin * 2 * pi * j * PlayableMatrix(i,3) * timeVector ) ;
    end
    waveVector = .5*waveVector/max(waveVector) ;
    dampingVector = cos(fpsin * pi * 1/(2*duration) * timeVector) ;
    
    for j = 1:floor(length(timeVector))
        if j < length(timeVector)/8
            dampingVector(j) = dampingVector(j)*0.5*(-cos(fpsin * 8 * pi * 1/duration * timeVector(j))+1 ) ;
        end
    end
    
    waveVector =  waveVector .* dampingVector ;
    start = PlayableMatrix(i,1) * samplesPerFrame ;
    musicVector(start:start+length(waveVector)-1) = musicVector(start:start+length(waveVector)-1) + waveVector ;
    
    if mod(i,rCeil60) == 0
        fprintf('-')
    end
    
end

out = musicVector ;

end