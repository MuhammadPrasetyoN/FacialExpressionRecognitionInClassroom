
# Facial Expression Recognition in Classroom

This project implements facial expression recognition using a semi-supervised learning model called `Progressive Teacher`. The model is designed to classify seven basic facial expressions: angry, disgust, fearful, happy, neutral, sad, and surprised. The project uses OpenCV for real-time video processing and the `YuNet` face detector for detecting faces.

## INSTALLATION

To set up the project, follow these steps:

1. **Clone the repository:**
```bash
git clone https://github.com/MuhammadPrasetyoN/FacialExpressionRecognitionInClassroom.git
cd FacialExpressionRecognitionInClassroom
```
2. **Go to the working directory**

    in folder FacialExpressionRecognitionInClassroom

3. **Install the required Python packages:** 
    
    Make sure you have Python 3.8 or higher. Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the pre-trained model:** 

    The pre-trained model is included in the repository, but if necessary, you can download it from [here](https://github.com/opencv/opencv_zoo/blob/main/models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx).

## USAGE

You can run the demo using either images or real-time video from your camera. Below are examples of how to use the script:

### Go to The Directory
```bash
search the demo.py file in folder FacialExpressionRecognitionInClassroom
```

### Run the Demo
 ```bash
 python demo.py
 ```

### How to Stop
The program will run continuously until you press the `P` key on your keyboard to stop it.

## HoOW THE PROGRAM WORKS
1. **Face Detection:**  
    The program uses the YuNet face detector to identify faces in the input image or video stream.

2. **Facial Expression Recognition:**  
    For each detected face, the model classifies the facial expression into one of the seven categories.

3. **Visualization:**  
    The results are drawn on the input image, including bounding boxes around faces and the recognized expressions.

4. **Data Logging:**    
    Detected faces and expressions are logged. The results for each detected image are saved in a folder, and the expression data are recorded in an Excel file.

## EXAMPLE OUTPUT

### Detected Image
Here’s an example of the output with recognized expressions:

![Detected Image](/FacialExpressionRecognitionInClassroom/detected_images/20240904_091625/example_detected_image.png)

### Excel File
The program also generates an Excel file containing detailed logs of the detected faces and their expressions.

![Data In Excel](/FacialExpressionRecognitionInClassroom//detected_images/20240904_091625/example_data_in_excell.png)

## LICENSE
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.