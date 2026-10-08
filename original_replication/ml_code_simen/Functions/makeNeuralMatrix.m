function [outMatrix, outVector] = makeNeuralMatrix( MidiMatrix )
%This function makes the matrix that will be analysed:
%It consists of multiple parts; 
%   1: The note pressed
%   2: The scale of the note
%   3: The duration of the note
%   4: The time at which the note starts
%   5: The hundred last notes played


[r, ~] = size(MidiMatrix) ;
Matrix = zeros(101,r) ;

for j = 1:r
    Matrix(1,j) = MidiMatrix(j,3) ; %Note
    Matrix(2,j) = mod(MidiMatrix(j,3),12) ; %Scale
    Matrix(3,j) = MidiMatrix(j,6) - MidiMatrix(j,5) ;%Duration
    Matrix(4,j) = MidiMatrix(j,5) ; %Time of start
    if j <= 100
        for i = 2:j
            Matrix(i+3,j) = MidiMatrix(j,3) - Matrix(1,j-i+1) ;
        end
    else
        for i = 2:101
            Matrix(i+3,j) = MidiMatrix(j,3) - Matrix(1,j-i+1) ;
        end
    end
end %Creating the hundred past notes. 

Matrix(:,r) = [] ; %Deleting the last row
Matrix(:,1:100) = [] ; %Deleting the first hundred rows.


NeuralVector = zeros(88, r-101) ; %Creating the vector
for i = 101:r-1
    for j = 1:88
        if j == MidiMatrix(i,3)
            NeuralVector(j,i-100) = 1 ;
        end
    end
end
outMatrix = Matrix ;
outVector = NeuralVector ;






end

