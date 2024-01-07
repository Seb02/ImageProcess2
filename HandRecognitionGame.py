import cv2
import numpy as np
import random


cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30)
redDot = cv2.imread('RedDot.png', 0)
redSquare = cv2.imread('RedSquare.png', 0)
Dotmask = cv2.cvtColor(redDot, cv2.COLOR_BGR2RGB)
Squaremask = cv2.cvtColor(redSquare, cv2.COLOR_BGR2RGB)
Dot = cv2.resize(Dotmask, (275, 275), interpolation=cv2.INTER_AREA)
Square = cv2.resize(Squaremask, (275, 275), interpolation=cv2.INTER_AREA)

shapeHeight, shapeWidth, _ = Dot.shape
print(shapeWidth, shapeHeight)
shapeHeight1, shapeWidth1, _ = Square.shape
print(shapeWidth1, shapeHeight1)
shapePosition = (1000, 50)
template1 = None
Score = 0
color = None

def capture_sample(frame, box_coords):
    x, y, w, h = box_coords
    return frame[y+6:y+h-6, x+6:x+w-6]

def find_match(frame, template):
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCORR_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    #best match
    top_left = max_loc
    h, w = template.shape[:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)

    contours, hierarchy = cv2.findContours(frame_gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_contour)

       
        mask_hull = frame_gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()

        
        cv2.drawContours(mask_hull, [hull], -1, (255, 255, 255), 2)
        mask = mask_hull
        number_of_edges = len(hull)
        

      
        
        
    else:
        mask = frame_gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        number_of_edges = 0

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return top_left, bottom_right, mask_bgr, number_of_edges


box_x, box_y, box_w, box_h = 100, 100, 300, 300  


cap = cv2.VideoCapture(0)



i = 0
chosenShape = 1
while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    ret, th = cv2.threshold(fgmask, 100, 200, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, None, iterations=4)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

   
    fgmask_colored = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

   
    cv2.drawContours(fgmask_colored, contours, -1, (0, 255, 0), 2)

    if not ret:
        break

   
    

    if template1 is None:
        cv2.rectangle(fgmask_colored, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
        if cv2.waitKey(1) & 0xFF == ord('c'):  # Press 'c' to capture the sample
            template1 = capture_sample(fgmask_colored, (box_x, box_y, box_w, box_h))
            print("Sample1 captured")

   

    if template1 is not None:
        top_left1, bottom_right1, mask1, edges = find_match(fgmask_colored, template1)
        res1 = cv2.matchTemplate(fgmask_colored, mask1, cv2.TM_CCOEFF_NORMED)
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
        

  


        if edges < 18:
            cv2.rectangle(fgmask_colored, top_left1, bottom_right1, (255, 255, 0), 2) #aqua
            fgmask_colored[top_left1[1]:bottom_right1[1], top_left1[0]:bottom_right1[0]] = mask1
            
            color = "aqua"
        if 17 < edges:
            cv2.rectangle(fgmask_colored, top_left1, bottom_right1, (255, 255, 255), 2) #white
            fgmask_colored[top_left1[1]:bottom_right1[1], top_left1[0]:bottom_right1[0]] = mask1
            color = "white"
       
    if i>20 and template1 is not None:
        print("debug")
        
        center_of_square = ((top_left1[0] + bottom_right1[0]) / 2, (top_left1[1] + bottom_right1[1]) / 2)

        shapePosition_center = (shapePosition[0] + 137, shapePosition[1] + 137)

        distance = ((shapePosition_center[0] - center_of_square[0])**2 + (shapePosition_center[1] - center_of_square[1])**2)**0.5

      
        
        if color == "aqua" and chosenShape == 0 and distance < 400 :
            Score +=1
        if color == "white" and chosenShape == 1 and distance < 400 :
            Score +=1
        shapePosition =  (random.randint(0, 1850), random.randint(0, 1000))

        if shapePosition[0] + shapeWidth > fgmask_colored.shape[1]:
            shapePosition = (50, shapePosition[1])

        if shapePosition[1] + shapeHeight > fgmask_colored.shape[0]:
            shapePosition = (shapePosition[0], 50)
        
        chosenShape = random.randint(0, 1)

        if chosenShape == 1:
            fgmask_colored[shapePosition[1]:shapePosition[1] + shapeHeight, shapePosition[0]:shapePosition[0] + shapeWidth] = Dot
        if chosenShape == 0:
            fgmask_colored[shapePosition[1]:shapePosition[1] + shapeHeight, shapePosition[0]:shapePosition[0] + shapeWidth] = Square
        
        i = 0
    if chosenShape == 1:
        fgmask_colored[shapePosition[1]:shapePosition[1] + shapeHeight, shapePosition[0]:shapePosition[0] + shapeWidth] = Dot
    if chosenShape == 0:
        fgmask_colored[shapePosition[1]:shapePosition[1] + shapeHeight, shapePosition[0]:shapePosition[0] + shapeWidth] = Square
        


    text = "Press 'c' to capture the sample, score = "
    
    
   
    cv2.putText(fgmask_colored, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(fgmask_colored, str(Score), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Template Matching', fgmask_colored)

    if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to quit
        break
    i += 1
    

