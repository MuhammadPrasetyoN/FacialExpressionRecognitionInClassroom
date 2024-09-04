import sys
import argparse
import copy
import datetime
import time
import os

import numpy as np
import cv2 as cv
import pandas as pd

from facial_fer_model import FacialExpressionRecog

sys.path.append('../face_detection_yunet')
from yunet import YuNet

# Check OpenCV version
assert cv.__version__ >= "4.9.0", \
       "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(description='Facial Expression Recognition')
parser.add_argument('--input', '-i', type=str,
                    help='Path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='./facial_expression_recognition_mobilefacenet_2022july.onnx',
                    help='Path to the facial expression recognition model.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--save', '-s', action='store_true',
                    help='Specify to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Specify to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()

def visualize(image, det_res, fer_res, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    print('%s %3d faces detected.' % (datetime.datetime.now(), len(det_res)))
    output = image.copy()
    #landmark_color = [
    #    (0,    0, 255),  # left eye
    #    (0,  255,   0),  # nose tip
    #    (255,  0, 255),  # right mouth corner
    #    (0,  255, 255)   # left mouth corner
    #]

    for ind, (det, fer_type) in enumerate(zip(det_res, fer_res)):
        bbox = det[0:4].astype(np.int32)
        fer_type = FacialExpressionRecog.getDesc(fer_type)
        print("Face %2d: %d %d %d %d %s." % (ind, bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], fer_type))
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
        cv.putText(output, fer_type, (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
        #landmarks = det[4:14].astype(np.int32).reshape((5, 2))
        #for idx, landmark in enumerate(landmarks):
        #    cv.circle(output, landmark, 2, landmark_color[idx], 2)
    return output

def classify_expression(fer_type):
    positive_classes = ['neutral', 'happy', 'surprised']
    if fer_type in positive_classes:
        return 'Positive'
    else:
        return 'Negative'

def process(detect_model, fer_model, frame):
    h, w, _ = frame.shape
    detect_model.setInputSize([w, h])
    dets = detect_model.infer(frame)

    if dets is None:
        return False, None, None

    fer_res = np.zeros(0, dtype=np.int8)
    for face_points in dets:
        fer_res = np.concatenate((fer_res, fer_model.infer(frame, face_points[:-1])), axis=0)
    return True, dets, fer_res

def capture_and_process(detect_model, fer_model):
    # Capture image from default camera
    deviceId = 0
    cap = cv.VideoCapture(deviceId)
    stop_program = False  # Variable to stop the program

    df_list = []

    # Create a directory to save the detected images
    save_dir = os.path.join('detected_images', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)

    while not stop_program:
        start_time = time.time()  # Get start time
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Get detection and fer results
        status, dets, fer_res = process(detect_model, fer_model, frame)

        if status:
            # Draw results on the input image
            frame = visualize(frame, dets, fer_res)

            # Save the detected image with HD quality
            image_name = f"detected_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            image_path = os.path.join(save_dir, image_name)
            cv.imwrite(image_path, frame, [cv.IMWRITE_JPEG_QUALITY, 95])  # Set JPEG quality to 95 for HD

            # Append data to the DataFrame
            df_list.append({'Time': datetime.datetime.now(), 
                            'Expression': fer_res,
                            'Total Detected Faces': len(dets),
                            'Total Detected Expressions': len(fer_res),
                            'Total Positive Expressions': sum([1 for fer_type in fer_res if classify_expression(FacialExpressionRecog.getDesc(fer_type)) == 'Positive']),
                            'Total Negative Expressions': sum([1 for fer_type in fer_res if classify_expression(FacialExpressionRecog.getDesc(fer_type)) == 'Negative'])
                            })

            # Visualize results in a new window
            cv.imshow('FER Demo', frame)
            key = cv.waitKey(3000)  # Display window for 10 seconds

            # Close window after 10 seconds
            cv.destroyAllWindows()

            if key == ord('P') or key == ord('p'):
                stop_program = True

        # Check if the user wants to exit (press 'P' key)
        if stop_program:
            break

        # Wait for the remaining time to maintain 60 seconds interval
        remaining_time = 10 - (time.time() - start_time)
        if remaining_time > 0:
            time.sleep(remaining_time)

    cap.release()

    # Create DataFrame from list of dictionaries
    df = pd.DataFrame(df_list)

    # Generate Excel filename with timestamp
    excel_filename = f"facial_expression_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    # Save DataFrame to Excel
    df['Expression'] = df['Expression'].apply(lambda x: ';'.join([FacialExpressionRecog.getDesc(fer_type) for fer_type in x]))
    df.to_excel(excel_filename, index=False)

    # Print the total expression results
    print("Total Expression Results:")
    for emotion, count in df['Expression'].str.split(';').explode().value_counts().items():
        print(f"{emotion}: {count} ({classify_expression(emotion)})")

    # Print the total positive and total negative expressions
    total_positive = df['Total Positive Expressions'].sum()
    total_negative = df['Total Negative Expressions'].sum()
    print(f"Total Positive Expressions: {total_positive}")
    print(f"Total Negative Expressions: {total_negative}")

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    detect_model = YuNet(modelPath='../face_detection_yunet/face_detection_yunet_2023mar.onnx')

    fer_model = FacialExpressionRecog(modelPath=args.model,
                                      backendId=backend_id,
                                      targetId=target_id)

    # If input is not specified, capture images from the default camera
    if args.input is None:
        capture_and_process(detect_model, fer_model)
    else:
        print("Input option not supported for real-time camera capture. Omit the --input option to use the camera.")
