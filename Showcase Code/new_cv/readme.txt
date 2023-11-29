This is the folder file where the new computer vision code for chess pieces without NAMCO ID tag is created.

So far it has:
- Cropping chessboard, allowing it to align with camera no matter which camera angle
- Bit masking to obtain black-white chessboard image

As the bit-masked image on the chessboard shows unclear difference for queen, bishop, and knight. Further development needs to be done.
Developing:
- Clear differentiation of individual chess pieces.(Trying out morph gradient and bit-masking).

Missing:
- Dividing chessboard squares into coordinates and converting to matrix
- edge detection for chess pieces
