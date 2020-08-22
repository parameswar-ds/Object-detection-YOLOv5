import argparse

import torch.backends.cudnn as cudnn

# from utils import google_utils
from utils.datasets import *
from utils.utils import *
import cv2

device=""
imgsz=416
# Initialize
device = torch_utils.select_device(device)
weights="weights/last_yolov5s_results.pt"
start = time.time()
model = torch.load(weights, map_location=device)['model'].float() 
end = time.time()
print(f"Runtime to load weights is {end - start}") 


c=0
a_v=0
def cal(l):
    global c
    global a_v
    x_c=(l[2]+l[0])/2
    y_c=(l[3]+l[1])/2
    if y_c>=400 and y_c<=600:
        lo=1
        if a_v!=lo:
            a_v=1
            c=c+1
    else:
        lo=0
        if a_v!=lo:
            a_v=0
    return c


def detect(img):
    global model
    global device
    global imgsz
    source=img
    conf_thres=0.4
    iou_thres=0.5
    
    
    half = device.type != 'cpu'
    # model.fuse()
    model.to(device).eval()
    print(imgsz)
    dataset = LoadImages(source, img_size=imgsz)
    
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        cv2.imwrite("dada.jpg",im0s)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t0 = time.time()
        pred = model(img)[0]
        # print(pred)
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        print(pred)
        t2 = time.time()
        print(f"Runtime for predict is {t2 - t0}") 

        # creating ROIs
        print(pred[0])
        det=pred[0]
        print(det[0][1])
        x_min=int(det[0][0])
        y_min=int(det[0][1])
        x_max=int(det[0][2])
        y_max=int(det[0][3])
        print(type(x_max))
        print("{}:{}:{}:{}".format(x_min,y_min,x_max,y_max))
        c=cal([x_min,y_min,x_max,y_max])
        # printing
        #im0=cv2.imread("2.jpg")
        cv2.rectangle(im0s,(x_min,y_min),(x_max,y_max),(0,255,0),2)
        cv2.line(im0s,(0,200),(416,200),(0,0,255),5)
        cv2.line(im0s,(0,300),(416,300),(0,0,255),5)
        cv2.putText(im0s, "count:"+str(int(c)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(im0s,"centroid",((x_max+x_min)//2,(y_max+y_min)//2),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imwrite('Features.jpg', im0s)
       
        




img="vcxvgsd.jpg"


o=detect(img)
print(o)