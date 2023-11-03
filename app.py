from flask import Flask, render_template, request, send_file
import cv2
import os

app = Flask(__name__)

# Function to perform object detection
def detect_objects(image_path):
    img = cv2.imread(image_path)

    with open(os.path.join("project_files", 'obj.names'), 'r') as f:
        classes = f.read().splitlines()

    net = cv2.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
    classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

    for (classId, score, box) in zip(classIds, scores, boxes):
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                      color=(0, 255, 0), thickness=2)

    result_path = "Test_Image/result.jpg"
    cv2.imwrite(result_path, img)
    return result_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save the uploaded image
        f = request.files['file']
        image_path = "Test_Image_1.jpg Test_Image_2.jpg"
        f.save(image_path)

        # Perform object detection
        result_path = detect_objects(image_path)
        return render_template("result.html", result_path=result_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
