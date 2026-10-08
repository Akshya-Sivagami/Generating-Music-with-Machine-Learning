%This is the script that will make the dictionary of chords and generate the new overfitted song. 

Chord = mod(MidiMatrixChord(:,3),12) ;
r = length(Chord) ;

%Retrieving the first three notes:
ThreeNotes = sort(Chord(1:3)) ;

%Now running through the notes
i = 4 ;
j = 1 ;
ChordChangeVector = [] ;
ChordMatrix = zeros(3, 500) ;

while i < r-3 %So running through the song
    NextThreeNotes = sort(Chord(i:i+2)) ;
    if sum(ThreeNotes == NextThreeNotes) == 3
        i = i + 1 ;
        continue
    else
        ChordChangeVector(j) = i  ;
        ChordMatrix(j,1:3) = NextThreeNotes + 1 ;
        j = j + 1 ;
        i = i + 3 ;
        
        ThreeNotes = NextThreeNotes ;
        
    end
end

MidiMatrixChord(:,2) = 2 ;
for i = 1:length(ChordChangeVector)
    MidiMatrixChord(ChordChangeVector(i), 2) = 3 ;
    MidiMatrixChord(ChordChangeVector(i), 7:9) = ChordMatrix(i,1:3) ;
end

ChordMatrix = unique(ChordMatrix(:,:), 'rows') ;

%Combining the two matrices:
CombinedMatrix = [MidiMatrixChord ; [MidiMatrixNote zeros(length(MidiMatrixNote), 3) ] ] ;

%Sorting the matrix_
CombinedMatrix = sortrows(CombinedMatrix, 5) ;

%Going through the combined matrix:
[row, kol] = size(ChordMatrix) ;
[r, k] = size(CombinedMatrix) ;

i = 1 ;
while i  < r
    
    if CombinedMatrix(i,2) == 3
        vector = CombinedMatrix(i,7:9) ;
        i = i+1 ;
        while CombinedMatrix(i,2) ~= 3
            if CombinedMatrix(i,2) == 1
                for j = 1:row
                    if sum(vector == ChordMatrix(j,1:3)) == 3
                        index = 3 ;
                        while ChordMatrix(j,index) ~= 0
                            index = index + 1;
                        end
                        ChordMatrix(j,index) = CombinedMatrix(i,3) ;
                    end
                end
            end
            i = i+1 ;
            if i > r
                break
            end
        end
    else
        i = i+ 1;
    end
end

% Random music generation:
i = 1 ;
while i  < r
    
    if CombinedMatrix(i,2) == 3
        vector = CombinedMatrix(i,7:9) ;
        i = i+1 ;
        while CombinedMatrix(i,2) ~= 3
            if CombinedMatrix(i,2) == 1
                for j = 1:row
                    if sum(vector == ChordMatrix(j,1:3)) == 3
                        guess = 0 ;
                        while guess == 0
                            guess = ChordMatrix(j,randi(400)+3) ;
                        end
                        CombinedMatrix(i,3) = guess ;
                        CombinedMatrix(i,2) = 4 ;
                    end
                end
            end
            i = i+1 ;
            if i > r
                break
            end
        end
    else
        i = i + 1 ;
    end
end
    
                            
            
  
Midi= matrix2midi(CombinedMatrix(:,1:6));

writemidi(Midi, 'VirginiaNew.mid');