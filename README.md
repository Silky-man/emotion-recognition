## Emotion recognition
* cnnFace_and_emotion.py: Real-time detection of faces and recognition of expressions.
* emotion.hdf5: Weight file.
* keras-EmotionNetwork.py: An expression recognition network without mmd(by keras).
* mmd.py: An expression recognition network with mmd(by tf).
* utils.py: Normalized  
## Using
1. pip3 install mtcnn.
2. adjust the weight file path in cnnFace_and_emotion.py.
3. run cnnFace_and_emotion.py
4. if you want to identify the picture  
    change cv2.videocapture() to cv2.imread().
	
