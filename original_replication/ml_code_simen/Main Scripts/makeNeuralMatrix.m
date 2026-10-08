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
    
    if j > 100
        for i = 5:104
            Matrix(i,j) = Matrix(1,j) - Matrix(1,j-i+4) ;
        end
    end
end %Creating the hundred past notes. 

Matrix(:,1:100) = [] ; %Deleting the first hundred rows.

[~,k] = size(Matrix) ;



NeuralVector = zeros(88, k-1) ; %Creating the vector
for i = 1:k-1
    for j = 1:88
        if j == Matrix(1,i+1)
            NeuralVector(j,i) = 1 ;
        end
    end
end
Matrix(:,k) = [] ; %Deleting the last column
outMatrix = Matrix ;
outVector = NeuralVector ;






end

