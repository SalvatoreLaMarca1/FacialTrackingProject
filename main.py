import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def main():
    cam = cv2.VideoCapture(0)
    
    # output = cv2.VideoWriter( 
    #     "output.avi", cv2.VideoWriter_fourcc(*'MPEG'),  
    #   30, (1080, 1920)) 

    while True:
        check, frame = cam.read()
        
        if check is False:
            break # terminate if could not read 
        
        # adds box to each frame
        faces = detect_bounding_box (frame) 

        cv2.imshow('webcam', frame)
        
        key = cv2.waitKey(1)
        if key == 27: # ESC key
            break

    cam.release()
    cv2.destroyAllWindows()
    
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

if __name__=="__main__": 
    main() 