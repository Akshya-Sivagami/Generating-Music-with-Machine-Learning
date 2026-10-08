function playMatrix( Matrix )
fps = 30 ;
durationFrames = max(Matrix(:,1)) ;
iterator = 1 ;
for i = 1:durationFrames - 1
    if i == Matrix(iterator,1)
        if (i ~= Matrix(iterator+1,1))
            playFrequencies(Matrix(iterator,3),(Matrix(iterator,2)/fps)) ;
            iterator = iterator + 1 ;
        else
            while i == Matrix(iterator,1)
                playFrequencies(Matrix(iterator,3),(Matrix(iterator,2)/fps)) ;
                iterator = iterator + 1 ;
            end
        end
    end 
    pause(1/(fps + 5)) ;
end

end