%This function will only be for writing a very simple MidiSong
 Vector = [64 66 68 64 66 68 66 64 63 61 63 64 66 63 59 64 66 68 64 66 68 66 64 63 61 66 63 59 64]' ;
 Vector = Vector ;
 Matrix = ones(length(Vector),6) ;
 
 
 half = floor(length(Vector)/2) ;
 Matrix(1:half,2) =  Matrix(1:half,2)*2 ;
 Matrix(:,3) = Vector ;
 Matrix(:,4) = Matrix(:,4)*120 ;
 
 for i = 1:length(Vector)
     Matrix(i,5) = (i-1)*0.5 ;
     Matrix(i,6) = i*0.5 ;
 end
 Midi= matrix2midi(Matrix);

writemidi(Midi, 'Try.mid');
fprintf('Complete! \n')

