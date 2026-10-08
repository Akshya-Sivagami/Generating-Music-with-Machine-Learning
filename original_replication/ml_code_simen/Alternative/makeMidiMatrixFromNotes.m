function out =  makeMidiMatrixFromNotes( notes, bpm, transpose )
%This function accepts the above statements and it spits out a valid note
%progression.

beatspersecond = bpm/60 ;
secondperbeat = 1/beatspersecond ;

Matrix = ones(length(notes), 6) ;
Matrix(:,3) = notes - 4 + transpose;
Matrix(:,4) = 80 ;
beat = 0.50 ;

iterator = 0 ;
for i = 1:length(notes)
    Matrix(i,5) = iterator*beat*secondperbeat ;
    Matrix(i,6) = Matrix(i,5) + secondperbeat*beat ;
    iterator = iterator + 1 ;
end

out = Matrix ;

end