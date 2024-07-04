import os
import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

#Load yaml file containing class labels
with open(r"C:\Users\PC\Desktop\Ma'am\YOLO\Datapreparation\data.yaml", mode='r') as f:
    data_yaml=yaml.load(f, Loader=SafeLoader)#load yaml files
labels=data_yaml['names'] #Extract class labels from YAML files

#Load YOLO model from ONNX format
yolo=cv2.dnn.readNetFromONNX(r"C:\Users\PC\Desktop\Ma'am\YOLO\Prediction\Model7\weights\best.onnx")
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#Open video path file for object detection
#video_path=r"C:\Users\PC\Desktop\Ma'am\YOLO\Prediction\video.mp4"
#cap=cv2.VideoCapture(video_path)

#Webcam access
cap=cv2.VideoCapture(0)

while True:
    ret, frame=cap.read()
    
    if not ret: #If no frmae is grabbed(end of video), break out of the loop
        break
    
    #preprocess frame and perform YOLO Object detection
    max_rc=max(frame.shape[:2]) #Get maximum dimension of the image
    input_image=np.zeros((max_rc, max_rc, 3), dtype=np.uint8) #Create a blank canvas
    input_image[0:frame.shape[0], 0:frame.shape[1]]=frame #paste frame onto the canvas
    INPUT_WH_YOLO=(640,640) #define input size for YOLO image
    blob=cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO), swapRB=True, crop=False)
    yolo.setInput(blob) #Set input for YOLO model
    preds= yolo.forward() #perform forward pass and get predictions
    
    #Preprocess detection and draw a bounding boxes
    detections=preds[0]
    boxes=[]
    confidences=[]
    classes=[]
    
    image_w, image_h = input_image.shape[:2]
    x_factor=image_w/INPUT_WH_YOLO[0]
    y_factor=image_h/INPUT_WH_YOLO[1]
    
    for i in range(len(detections)):
        row=detections[i]
        confidence=row[4]
        if confidence>0.4:
            class_score=row[5:].max()
            class_id=row[5:].argmax()
            if class_score>0.25:
                cx, cy, w, h=row[0:4]
                left=int((cx-0.5*w)*x_factor)
                top=int((cy-0.5*h)*y_factor)
                width=int(w*x_factor)
                height=int(h*y_factor)
                box=np.array([left, top, left+width, top+height])
                
                confidences.append(class_score)
                boxes.append(box)
                classes.append(class_id)
                
    boxes_np=np.array(boxes)
    confidences_np=np.array(confidences)
    
    #Perform non-maximum supression to remove redundant bounding boxes
    output=cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)
    if len(output)>0:
        index=output.flatten()
    else:
        index=np.empty((0), dtype=int)
    
    #Draw bounding boxes and labels on the frame
    for ind in index:
        x,y,w,h=boxes_np[ind]
        bb_conf=int(confidences_np[ind]*100)
        class_id=classes[ind]
        class_name=labels[class_id]
        
        text=f'{class_name}:{bb_conf}%'
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.rectangle(frame, (x,y-30), (x+w, y), (255,255,255),-1)
        cv2.putText(frame, text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    #Display the frame with boundary box
    cv2.imshow("YOLO OBJECT DETECTION", frame)
    
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()