import cv2
from ultralytics import YOLO
import os

# upload model
model = YOLO("best.pt")

# image path
image_dir = "C:/Users/90545/Desktop/image"

#Get all image paths in directory
image_paths = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)
               if image_name.endswith(('.png', '.jpg', '.jpeg'))]

for image_path in image_paths:
    #read and resize image
    image = cv2.imread(image_path)
    image=cv2.resize(image,(640,640))

    if image is None:
        print(f"{image_path} image cannot read.")
        continue

    #Detect and track objects with YOLOv8
    results = model(image)

    #Process each detection result
    for result in results:
        #plot result
        annotated_image = result.plot()

        #show plotting image
        cv2.imshow("YOLOv8 Tracking", annotated_image)

        key = cv2.waitKey(0)

        # press q finish loop
        if key == ord('q'):
            break

#close all window
cv2.destroyAllWindows()