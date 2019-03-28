from util.face_roi import *
import tqdm
import time
model = faceROI()
cap = cv2.VideoCapture('examples/input.mp4')
while(True):
    s = time.time()
    ret, frame = cap.read()
    faces, roi = model.detect(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # print(1)
    image = model.fuse(frame, roi, faces)
    cv2.imshow('a', image)
    e = time.time()
    print(e-s)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

