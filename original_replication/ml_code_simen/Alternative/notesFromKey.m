function out = notesFromKey( letter )
ShapeName=char(letter);
switch ShapeName
    case char('C')
        vector = [40 44 40 47 40 44 40 47] ;
    case char('F')
        vector = [40 45 40 49 40 45 40 49] ;
    case char('G')
        vector = [39 42 39 47 39 42 39 47] ;
    case char('Am')
        vector = [40 44 40 49 40 44 40 49] ;
    case char('Dm')
        vector = [42 45 42 49 42 45 42 49] ;
    case char('G7')
        vector = [39 45 39 47 39 45 39 47] ;
    case char('Em')
        vector = [39 44 39 47 39 44 39 47] ;
    otherwise
        disp('Error, no such chord is found! Try again!')
end
out = vector ;


end