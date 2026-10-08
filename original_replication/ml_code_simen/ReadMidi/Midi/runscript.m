[y,Fs] = midi2audio('TTLS.mid');    

%% listen in matlab:
soundsc(y, Fs);  % FM-synth