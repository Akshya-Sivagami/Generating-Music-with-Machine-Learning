

Matrix = getFrequencies('Faded.mp4',50) ;
    
v = VideoReader('Faded.mp4');
imageClean = read(v,50) ;
imshow(read(v,69))

playFromImage(imageClean,read(v,69))