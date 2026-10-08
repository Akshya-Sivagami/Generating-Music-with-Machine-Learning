%This will make a chord progression 

bpm = 80 ;
transpose =  12 ;
notes = stitchNotes('C G Am F C G Am F C G Am F' ) ;
%Making the midi-matrix:

MidiMatrix = makeMidiMatrixFromNotes(notes, bpm, transpose) ;

%Making the file
Midi= matrix2midi(MidiMatrix);
writemidi(Midi, 'TryChord.mid');
fprintf('Complete! \n')