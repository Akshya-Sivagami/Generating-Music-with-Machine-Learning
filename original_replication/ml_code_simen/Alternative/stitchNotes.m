function out = stitchNotes( ChordSequence )
%This spits out the long series of notes to play:
series = [] ;

array = strsplit(ChordSequence) ;
for i = 1:length(array)
    series = [series, notesFromKey(array(i))] ;
end


out = series ;

end