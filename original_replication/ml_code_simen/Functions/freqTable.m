function [ out ] = freqTable( allVector, AVector )
%This function accepts the vector of all keys, and the vector of all A's 
%It then returns a vector with length equal to allVector with the relevant
%frequencies.
startingA = 37  ;


freqVector = 0 ;
for i = 1:12
    if allVector(i) == AVector(1)
        freqVector(i) = startingA ;
        j = i ;
    end
end

for i = 1:length(allVector)-j
    freqVector(i+j) = startingA + i ;
    %freqVector(i+j) = startingA*2^(i/12) ;
end

if j > 1
    for i = 1:1:j-1
        freqVector(j-i) = startingA - i ;
        %freqVector(i) = (startingA/2)*2^(i/12) ;
    end
end


out = freqVector    ;

%Her skriver vi ut nummeret på der vi skal taste