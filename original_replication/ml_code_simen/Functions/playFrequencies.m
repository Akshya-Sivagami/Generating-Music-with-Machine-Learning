function  playFrequencies( frequencies, duration )
%This will give the sound of the frequencies;
dampedWave = 0 ;
for j = 1:length(frequencies)
    
    samplingFrequency = 44100 ;
    soundVector = [ 1 0.5 0.5 0.5 ] ;
    timeVector = linspace(1/samplingFrequency, duration, duration*samplingFrequency) ;
    waveVector = zeros(1,floor(samplingFrequency*duration)) ;
    
    for i = 1:length(soundVector)
        waveVector = waveVector + soundVector(i) * cos( 2 * pi * i * frequencies(j) * timeVector ) ;
    end
    waveVector = .5*waveVector/max(waveVector) ;
    dampingVector = cos(   pi * 1/(2*duration) * timeVector) ;
    
    for i = 1:floor(length(timeVector))
        if i < length(timeVector)/8
            dampingVector(i) = dampingVector(i)*0.5*(-cos(8 * pi * 1/duration * timeVector(i))+1 ) ;
        end
    end
    dampedWave = dampedWave + waveVector .* dampingVector ;
%     plot(timeVector, dampedWave)
%     soundsc( dampedWave, samplingFrequency )
    
end

soundsc( dampedWave, samplingFrequency )







end

