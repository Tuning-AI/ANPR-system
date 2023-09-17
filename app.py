from ultralytics import YOLO
import cv2
import cvzone
import math
import streamlit as st
from lib.ocr_fun import ocr
import numpy as np 
import time 
from lib.sort import *
import csv
import pandas as pd 
ocro = ocr()
area = [(1018, 382)
,(1019, 473)
,(1019, 476)
,(1016, 533)
,(1016, 532)
,(596, 514)
,(565, 514)
,(417, 514)
,(244, 503)
,(66, 506)
,(1, 509)
,(1, 483)
,(0, 370)
,(0, 345)]

lplist = []
counter = 0

with st.sidebar : 
    st.image("icon.png" , width=150)
    select_type_detect = st.selectbox("Detection from :  ",
                                            ("File", 
                                             "Live"))
    save_lpcrops = st.selectbox("Do you want to save license plate Crops ? " , 
                             ("Yes" , "No"))
    save_cacrops = st.selectbox("Do you want to save Car Crops ? " , 
                             ("Yes" , "No"))
    select_device = st.selectbox("Select compute Device :  ",
                                            ("CPU", "GPU"))
    save_output_video = st.radio("Save output video?",
                                            ('Yes', 'No'))
    confd = st.slider("Select threshold confidence value : " , min_value=0.1 , max_value=1.0 , value=0.25)
    iou = st.slider("Select Intersection over union (iou) value : " , min_value=0.1 , max_value=1.0 , value=0.5)

tab1 , tab2= st.tabs(["Detection" , "Log"])
with tab1 : 
    if select_device == "GPU" : 
        DEVICE_NAME = st.selectbox("Select GPU index : " , 
                                     (0, 1 , 2)) 
    elif select_device =="CPU" : 
        DEVICE_NAME = "cpu"
    fpsReader = cvzone.FPS()
    class_names = ["license-plate", "vehicle"] 
    if select_type_detect == "File" : 
        file = st.file_uploader("Select Your File : " ,
                                 type=["mp4" , "mkv"])
        if file : 
            source = file.name
            cap = cv2.VideoCapture(source)
    elif select_type_detect == "Live" : 
        source = st.text_input("Past Your Url here and Click Entre")
        cap = cv2.VideoCapture(source)
    # creat the model
    car_tracker = Sort(max_age=20)
    lp_tracker = Sort(max_age=20)
    model = YOLO("models/license-plates-us-eu-zgfga.pt")
    frame_window = st.image( [] )
    col_car , col_lp = st.columns(2)
    start , stop = st.columns(2)
    with start :
        start = st.button("Click To Start")
    with stop : 
        stop = st.button("Click To Stop" , key="ghedqLKHF")
    if start :
        with col_car :
            st.write("The Detection car : ")
            car_frame = st.image([])
        with col_lp :
            st.write("The Detection License-plates : ")
            lp_frame = st.image([])
            lptext = st.markdown("0")
        while True :
            try : 
                _ , img = cap.read()
                img = cv2.resize(img, (1020,600))
                if save_output_video == "Yes" :
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
                    fourcc = cv2.VideoWriter_fourcc(*'MP4V') #use any fourcc type to improve quality for the saved video
                    out = cv2.VideoWriter(f'results/{source.split(".")[0]}.mp4', fourcc, 10, (w, h)) #Video settings
                # fps counter
                fps, img = fpsReader.update(img,pos=(20,50),
                                            color=(0,255,0),
                                            scale=2,thickness=3)
                if counter % 10 == 0 :
                    # make the prediction
                    results = model(img ,conf=confd ,
                                    iou=iou,
                                    device=DEVICE_NAME)
                    for result in results : 
                        # depackage results
                        bboxs = result.boxes 
                        for box in bboxs :
                            # bboxes
                            x1  , y1 , x2 , y2 = box.xyxy[0]
                            x1  , y1 , x2 , y2 = int(x1)  , int(y1) , int(x2) , int(y2)
                            #cv2.polylines(img,[np.array(area,np.int32)],True,(0,255,0),2)
                            # confidence 
                            conf = math.ceil((box.conf[0] * 100 ))
                            # class name
                            clsi = int(box.cls[0])
                            # calculate the width and the height
                            w,h = x2 - x1 , y2 - y1
                            # convert it into int
                            w , h = int(w) , int(h)
                            # draw our bboxes 
                            cvzone.cornerRect(img ,(x1 , y1 , w , h) ,l=7)
                            #if clsi == 1 : 
                            cvzone.putTextRect(img , f"{class_names[clsi]}" ,
                                                (max(0,x1) , max(20 , y1)),
                                                thickness=1 ,colorR=(0,0,255) ,
                                            scale=0.9 , offset=3)
                            if clsi == 1 :
                                in_region = cv2.pointPolygonTest(np.array(area,np.int32),(x2,y2),False)
                                if in_region >=0:
                                    crop_car = img[y1 : y1+h , x1:x1+w] 
                                    inr = True
                                    detections_car = np.empty((0,5)) 
                                    track_listcar = np.array([x1 , y1 , x2 , y2 , conf])
                                    detections_car = np.vstack((detections_car , track_listcar))
                                    result_track = car_tracker.update(detections_car)
                                    for bbox1 in result_track:
                                        xca1,yca1,xca2,yca2,idca = bbox1
                                    if save_cacrops == "Yes" :
                                        cv2.imwrite(f'Cars_nump/{int(idca)}.jpg', crop_car)
                            if clsi == 0 :
                                try :
                                    lp_crop = img[y1 : y1+h , x1:x1+w]
                                    track_lp = np.array([x1 , y1 , x2 , y2 , conf])
                                    detections_lp = np.empty((0,5))
                                    detections_lp = np.vstack((detections_lp , track_lp))
                                    cx=int(x1+x2)//2
                                    cy=int(y1+y2)//2
                                    #img = cv2.circle(img , (cx, cy) , 20 , (255, 0, 0) , 1)
                                    in_area = cv2.pointPolygonTest(np.array(area,np.int32),(cx,cy),False)
                                    if in_area >=0 and inr == True:
                                        resultlp_track = lp_tracker.update(detections_lp)
                                        #in_area = cv2.pointPolygonTest(np.array(area,np.int32),(cx,cy),False) 
                                        #text_result = ocro.easyocr_fun(lp_crop)
                                        #text_result = ocro.pytesseract_fun(crop)
                                        #text_result = ocro.AnPr(lp_crop)
                                        #put text_result inside our image
                                        #cvzone.putTextRect(img , f"{text_result}" ,
                                        #            (max(0,x1) , max(20 , y1)),
                                        #            thickness=1 ,colorR=(0,0,255) ,
                                        #            scale=0.9 , offset=3)
                                        for bbox2 in resultlp_track:
                                            xlp1,ylp1,xlp2,ylp2,idlp = bbox2  
                                        if save_lpcrops == "Yes" : 
                                            cv2.imwrite(f'num_plate/{int(idlp)}.jpg', lp_crop)
                                    
                                        text_result = ocro.easyocr_fun(lp_crop)
                                        
                                        lptext.write(f"The License-plates Number  :  {text_result}")
                                        
                                        lplist.append(text_result)

                                        countlp = lplist.count(text_result)
                                        if countlp == 4 : 
                                            with open('logs/log.csv', 'a') as f:
                                                date_time = str(time.asctime())
                                                header = [text_result,date_time ]
                                                # create the csv writer
                                                writer = csv.writer(f)
                                                # write a row to the csv file
                                                writer.writerow(header)
                                                f.close()
                                                lplist.clear()

                                        #crop_car  = cv2.cvtColor( crop_car , cv2.COLOR_BGR2RGB)
                                        car_frame.image(f'Cars_nump/{int(idca)}.jpg')

                                        #lp_crop  = cv2.cvtColor( lp_crop , cv2.COLOR_BGR2RGB)
                                        lp_frame.image(f'num_plate/{int(idlp)}.jpg')
                            
                                except : 
                                    pass
                            else : 
                                pass
                            try: 
                                out.write(img)
                            except:
                                pass
                    frame  = cv2.cvtColor( img , 
                                            cv2.COLOR_BGR2RGB)        
                    frame_window.image(frame)
            except : 
                pass
            counter = counter + 1
    if stop:
        try:
            cap.release()
            
        except : 
            pass

with tab2 : 
    dataf = pd.read_csv("logs/log.csv")
    st.dataframe(dataf)


