from sre_constants import SUCCESS
import cv2 as cv
import numpy as np
from math import atan, degrees
import json
import time
import serial
ser=serial.Serial('COM11',9600)
def canny(img):
    if img is None:
        cap.release()
        cv.destroyAllWindows()
        exit()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv.Canny(gray, 50, 150)
    return canny

def red_white_masking(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    lower_y = np.array([10, 130, 120], np.uint8)
    upper_y = np.array([40, 255, 255], np.uint8)
    mask_y = cv.inRange(hsv, lower_y, upper_y)
    cv.namedWindow("mask_y", cv.WINDOW_NORMAL)
    cv.resizeWindow("mask_y", 500, 500)
    cv.imshow('mask_y', mask_y)
    
    lower_w = np.array([0, 0, 212], np.uint8)
    upper_w = np.array([170, 200, 255], np.uint8)
    mask_w = cv.inRange(hsv, lower_w, upper_w)
    cv.namedWindow("mask_w", cv.WINDOW_NORMAL)
    cv.resizeWindow("mask_w", 500, 500)
    cv.imshow('mask_w', mask_w)
    
    mask = cv.bitwise_or(mask_w, mask_y)
    cv.namedWindow("mask", cv.WINDOW_NORMAL)
    cv.resizeWindow("mask", 500, 500)
    cv.imshow('mask', mask)
    
    masked_bgr = cv.bitwise_and(image, image, mask=mask)
    cv.namedWindow("masked_bgr", cv.WINDOW_NORMAL)
    cv.resizeWindow("masked_bgr", 500, 500)
    cv.imshow('masked_bgr', masked_bgr)
    
    return masked_bgr


def filtered(image):
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    filtered_image = cv.filter2D(image, -1, kernel)
    cv.namedWindow("filtered_image", cv.WINDOW_NORMAL)
    cv.resizeWindow("filtered_image", 500, 500)
    cv.imshow('filtered_image', filtered_image)
    return filtered_image


def roi(image, vert, color=[255, 255, 255]):
    mask = np.zeros_like(image)
    cv.fillConvexPoly(mask, cv.convexHull(vert), color)
    masked_image = cv.bitwise_and(image, mask)
    cv.namedWindow("roi", cv.WINDOW_NORMAL)
    cv.resizeWindow("roi", 500, 500)
    cv.imshow('roi', masked_image)
    return masked_image


def edge_detection(image):
    edges = cv.Canny(image, 80, 200)
    cv.namedWindow("edges", cv.WINDOW_NORMAL)
    cv.resizeWindow("edges", 500, 500)
    cv.imshow('edges', edges)
    return edges


def average_slope_intercept(image, lines):
    left_fit = []    # list for all multiple lines found in left lane
    right_fit = []   # list for all multiple lines found in right lane
    global l
    global r
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    if left_fit != []:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = create_coordinates(image, left_fit_average)
        l = left_fit_average
    else:
        left_line = create_coordinates(image, l)
    if right_fit != []:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = create_coordinates(image, right_fit_average)
        r = right_fit_average
    else:
        right_line = create_coordinates(image, r)
    
    return np.array([left_line, right_line])


def create_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (2 / 3))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def draw_lines(left, right, image):
    cv.line(image, (left[0], left[1]), (left[2], left[3]), (0, 255, 0), 5)
    cv.line(image, (right[0], right[1]), (right[2], right[3]), (0, 255, 0), 5)
    vert = np.array([[(left[0], left[1]), (left[2], left[3]),
                    (right[0], right[1]), (right[2], right[3])]], np.uint64)
    cropped_lane = roi(image, vert, color=[0, 0, 255])
    detected_image = cv.addWeighted(image, 1, cropped_lane, 0.7, 0)
    
    cv.namedWindow("lines", cv.WINDOW_NORMAL)
    cv.resizeWindow("lines", 500, 500)
    cv.imshow('lines', detected_image)
    return detected_image

def process(image):
    h = image.shape[0]
    w = image.shape[1]

    masked = red_white_masking(image)
    blurred = filtered(masked)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    vertices = np.array([[(0, h), (w/2, h/2), (w, h)]], np.uint64)
    region_of_interest = roi(gray, vert=vertices)
    edges = edge_detection(region_of_interest)
    cv.imshow('edges', edges)

    lines = cv.HoughLinesP(edges, 1, np.pi/180, 20,
                           minLineLength=5, maxLineGap=200)

    left_lane, right_lane = average_slope_intercept(image, lines)
    final_image = draw_lines(left_lane, right_lane, image.copy())
    
    # Calculate the center of the lanes
    center_of_lanes = (left_lane + right_lane) / 2
    
    # Get the center of the frame
    center_of_frame = image.shape[1] // 2
    
    # Calculate the deviation of the car from the lane center
    deviation = center_of_frame - center_of_lanes[0]
    
    # Calculate the angle of deviation (in degrees)
    angle_degrees = degrees(atan(deviation / center_of_frame))
    
    # Determine the direction to move based on the deviation
    if deviation < 0:
        direction = "R"
    else:
        direction = "L"
    
     # Add text indicating the direction on the video frame
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(final_image, f"Move {direction} ({abs(angle_degrees)} degrees)", (50, 50), font, 1, (0, 255, 0), 2, cv.LINE_AA)
    if(angle_degrees>10):
        cv.putText(final_image, f"Warning!!! Car out of Lane", (50, 50), font, 1, (0, 0, 255), 2, cv.LINE_AA)
    
    
    print(f"Move {direction} with a deviation of {abs(angle_degrees)} degrees")
    json_data={"r":direction, "b":(abs(angle_degrees))}
    raj=json.dumps(json_data) + "*"
    ser.write(raj.encode())
    print (json_data)
    return final_image

cap = cv.VideoCapture('video5.mp4')

while SUCCESS:
    ret, frame = cap.read()

    if ret == False:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        _, frame = cap.read()

    detected = process(frame)
    
    cv.namedWindow("feed", cv.WINDOW_NORMAL)
    cv.resizeWindow("feed", 500, 500)
    cv.imshow('feed', frame)
    
    cv.namedWindow("detected lanes", cv.WINDOW_NORMAL)
    cv.resizeWindow("detected lanes", 500, 500)
    cv.imshow('detected lanes', detected)

    canny_image_1 = canny(frame)
    cv.namedWindow("canny_full", cv.WINDOW_NORMAL)
    cv.resizeWindow("canny_full", 500, 500)
    cv.imshow("canny_full", canny_image_1)

    if cv.waitKey(10) == 27:
        break
# ser.close(#include <ArduinoJson.h>

cv.destroyAllWindows()
cap.release()