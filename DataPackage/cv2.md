### face CascadeClassifier
```python
data_file=r"E:\Program Files\Python\Data\haarcascade_frontalcatface-alt.xml"
face_detector = cv.CascadeClassifier(data_file)  
faces = face_detector.detectMultiScale(imgG)
=>faces[face[x,y,w,h],]
```