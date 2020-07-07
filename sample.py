import cv2
import numpy as np
from gpiozero import CPUTemperature
from time import time
from detector import MotionDetector
from packer import pack_images
from datetime import datetime

def filter_fun(b):
    return ((b[2] - b[0]) * (b[3] - b[1])) > 1000
t=0
saved_time=0
lock1=0

if __name__ == "__main__":
   
    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)      
    detector = MotionDetector(bg_history=20,
                              brightness_discard_level=25,
                              bg_subs_scale_percent=0.1,
                              group_boxes=True,
                              expansion_step=5)
    now = datetime.now()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output ="backup/backup_" + now.strftime("%d""." "%m" "." "%Y" "_" "%X") +".avi"
    print(output)
    out = cv2.VideoWriter(output,fourcc, 20.0, (640,480))
    # group_boxes=True can be used if one wants to get less boxes, which include all overlapping boxes

    b_height = 512
    b_width = 512

    res = []
    while True:
        dt = datetime.now()
        
        cpu = CPUTemperature()
        
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break
        begin = time()
        boxes = detector.detect(frame)

        frame = cv2.putText(frame, dt.strftime("%x" " " "%X"),(0, 475),cv2.FONT_HERSHEY_DUPLEX, 1,(0, 255, 0),1, cv2.LINE_8) 
        # boxes hold all boxes around motion parts

        ## this code cuts motion areas from initial image and
        ## fills "bins" of 512x512 with such motion areas.
        ##
        # results = []
        # if boxes:
        #     results, box_map = pack_images(frame=frame, boxes=boxes, width=b_width, height=b_height,
        #                                    box_filter=filter_fun)
            # box_map holds list of mapping between image placement in packed bins and original boxes

        ## end

        for b in boxes:
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)

        end = time()
        it = (end - begin) * 1000
        res.append(it)
        print("StdDev: %.4f" % np.std(res), "Mean: %.4f" % np.mean(res), "Last: %.4f" % it, "Boxes found: ", len(boxes), "CPU_temp: ",cpu.temperature)

        idx = 0
        # for r in results:
        #     idx += 1
        #     cv2.imshow('packed_frame_%d' % idx, r)

        cv2.imshow('last_frame', frame)
        cv2.imshow('detect_frame', detector.detection_boxed)
        cv2.imshow('diff_frame', detector.color_movement)
        if t>0:
            actual_time = int(round(time() * 1000))
            if actual_time - saved_time>=1000: 
                t -= 1
                saved_time=actual_time
                print(t)


        if len(boxes) > 0:
            t=15
        if t>0:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()