function out = generateMidiMatrix( PlayableMatrix, fps, transpose )
fprintf('Generating Midi-Matrix... \n')
offcast = -4 + transpose ;
MidiMatrix = zeros(6) ;

[r, k] = size(PlayableMatrix)  ;
disp('Hallo')
iterator = 1 ;
check = 0 ;

for j = 2:k
    for i = 1:r-1
        if PlayableMatrix(i,j) < 1
            continue
        end
        check = 0 ;
        
        for p = 2:k
            if PlayableMatrix(i,j) == PlayableMatrix(i-1,p)
                check = 1 ;
                break
            end
        end
        
        if check == 0
            %In the case that a tone starts
            MidiMatrix(iterator,5) = PlayableMatrix(i,1)/fps ;
            %We set the starting moment
            MidiMatrix(iterator,3) = PlayableMatrix(i,j) + offcast;
            %We set the note
            future = 1 ;
            checkEnd = 0 ;
            while (checkEnd == 0) && (i + future < r )
                checkEnd = 1 ;
                for p = 2:k
                    if PlayableMatrix(i,j) == PlayableMatrix(i+future,p)
                        checkEnd = 0 ;
                        future = future + 1;
                        break
                    end
                end
            end
            MidiMatrix(iterator,6) = PlayableMatrix(i+future,1)/fps ;
            %Meta:
            MidiMatrix(iterator,1) = 1 ;
            MidiMatrix(iterator,2) = 1 ;
            MidiMatrix(iterator,4) = 120 ;
            iterator = iterator + 1 ;
            
            %Here we can toggle the tracks:
            
        end
    end
end


%Here we tidy up the matrix:
MidiMatrix = sortrows(MidiMatrix,5) ;

clc
fprintf('Generating Midi-Matrix Complete \n')
out = MidiMatrix(all(MidiMatrix,2),:);
end