from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torchvision
import os
from PIL import Image
import cv2
import io
import numpy as np
from paddleocr import PaddleOCR
app = FastAPI()
from autocorrect import Speller
import pytesseract
import matplotlib.pyplot as plt

n_classes=10
classes = [
    'A',
    'AC_Payee',
    'Account_number',
    'Amount',
    'Date',
    'MICR',
    'Payee',
    'Signature_1',
    'Signature_2',
    'Sum'
]

text_classes=["AC_Payee","Payee","Sum"]
number_classes=["Account_number","MICR","Amount","Date"]
# load an instance segmentation model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
print(in_features)
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)
model_path="E:\QuickFox\OCR\OCR_API\models\cheque_object_detection.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
model.to(DEVICE)
model.load_state_dict(torch.load(model_path,map_location=torch.device(DEVICE)))

model.eval()
torch.cuda.empty_cache()


ocr = PaddleOCR(use_angle_cls=True, lang='en',denoise={"method": "median"}, rectify="EAST")

pytesseract.pytesseract.tesseract_cmd = r'E:/QuickFox/OCR/OD Api/Tesseract-OCR/tesseract.exe'
config = ('-l eng --oem 3 --psm 6')

def process_image(img_bytes: bytes):
    try:
        # Convert bytes to OpenCV image
        print("Processing image")
        image = Image.open(io.BytesIO(img_bytes))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (1366, 768))

        # Convert to PyTorch tensor
        img_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return img_tensor
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")


def ocr_process(image_tensor,pred):
    image_tensor = (image_tensor.to(torch.float32) * 255.0).to(torch.uint8)
    spell = Speller(lang='en')
    object_images = []
    for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
        print(f"box: {box} label: {label} score: {score}")
        if score > 0.8:  # Adjust the threshold as needed
            # Extract the object from the image using the bounding box
            x1, y1, x2, y2 = box.int().tolist()
            # object_img = img_int[1, y1:y2, x1:x2]

            object_img = (image_tensor[1,y1:y2, x1:x2]).to('cpu').numpy()
            object_img = cv2.resize(object_img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_AREA)
            # img = cv2.threshold(object_img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
            # if classes[label] == "Sum" or classes[label] == "Payee":
            object_img = cv2.convertScaleAbs(object_img, alpha=1, beta=10)
            text = pytesseract.image_to_string(object_img, config=config)
            # print(f"text:{text}")
            results = ocr.ocr(object_img, det=False, cls=True, rec=True)
            # print(f"results: {results}")
            # print(f"result: {results[0][0]}")
            result = spell(results[0][0])
            # print(f"result spell: {result}")
            # print(f"label: {classes[label]}")
            # print(f"type: {type[result]}")

            # Append the object image to the list
            object_images.append((object_img, classes[label], result))
    return object_images


def crop_image(image_tensor,pred,img_name):
    image_tensor = (image_tensor.to(torch.float32) * 255.0).to(torch.uint8)
    # spell = Speller(lang='en')
    # object_images = []
    for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
        class_name=classes[label]
        print(f"box: {box} label: {label} score: {score}")
        if score > 0.8:  # Adjust the threshold as needed
            # Extract the object from the image using the bounding box
            x1, y1, x2, y2 = box.int().tolist()
            # object_img = img_int[1, y1:y2, x1:x2]

            object_img = (image_tensor[1,y1:y2, x1:x2]).to('cpu').numpy()
            object_img = cv2.resize(object_img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_AREA)

            # Specify the path where you want to save the resized image
            if class_name in text_classes:
                output_dir = r"E:\QuickFox\OCR\OCR_API\save\text\\"
            elif class_name in number_classes:
                output_dir = r"E:\QuickFox\OCR\OCR_API\save\number\\"
            else:
                continue
                # output_path = output_dir+"\\"+img_name[:-4]+"-"+class_name+".jpg"
            # output_dir = r"E:\QuickFox\OCR\OCR_API\save\\"
            output_path = output_dir+img_name[:-4]+"-"+classes[label]+".jpg"
              # Replace with the desired file path and format
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            # Save the resized image
            # img = cv2.threshold(object_img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
            # if classes[label] == "Sum" or classes[label] == "Payee":
            object_img = cv2.convertScaleAbs(object_img, alpha=1, beta=10)
            cv2.imwrite(output_path, object_img)
            # break 
    return
    
    

def get_result(object_images):
    results = {}
    for object_img, object_class, result in object_images:
        results[object_class] = result
    return results

def plot_result(object_images):
    # Calculate the number of rows and columns for subplots
    num_objects = len(object_images)
    num_cols = 3  # You can adjust the number of columns as needed
    num_rows = (num_objects + num_cols - 1) // num_cols
    print(f"num_rows: {num_rows} num_cols: {num_cols} num_objects: {num_objects}")
    # Create a figure with subplots for each detected object
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    results={}
    for i, (object_img, object_class, result) in enumerate(object_images):
        print(f"i: {i} object_class: {object_class} result: {result}")
        row = i // num_cols
        col = i % num_cols
        if num_rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]

        ax.imshow(object_img)
        
        ax.set_title(f"Class: {object_class}\n Text: {result}")
        ax.axis('off')
        results[object_class]=str(result)

    # Hide any empty subplots
    # print(results)
    for i in range(num_objects, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        if num_rows > 1:
            fig.delaxes(axes[row, col])
        else:
            fig.delaxes(axes[col])

    plt.show()
    return results

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    # try:
        # Read the content of the uploaded file
    img_bytes = await file.read()

    # Process the image using the provided function
    img_tensor = process_image(img_bytes)
    # all_results = []
    with torch.no_grad():
        img_tensor = img_tensor.to(DEVICE)  # Move the tensor to the device
        prediction = model([img_tensor])
        pred = prediction[0]
        print(prediction)
   
        object_images = ocr_process(img_tensor,pred)
        results = get_result(object_images)
        
    print(results)
    # Perform additional processing using img_tensor
    # (Add your custom processing logic here)

    return JSONResponse(content=results, status_code=200)
    # except ValueError as ve:
    #     return JSONResponse(content={"message": str(ve)}, status_code=400)
    # except Exception as e:
    #     return JSONResponse(content={"message": "Error processing file", "error": str(e)}, status_code=500)


@app.get("/")
async def read_user_item(
   
):
    item = {"Welcome to homepage:" }
    
    return item


@app.post("/getobjects/")
async def get_file(file: UploadFile = File(...)):
    # try:
        # Read the content of the uploaded file
    img_bytes = await file.read()

    # Process the image using the provided function
    img_tensor = process_image(img_bytes)
    # all_results = []
    with torch.no_grad():
        img_tensor = img_tensor.to(DEVICE)  # Move the tensor to the device
        prediction = model([img_tensor])
        pred = prediction[0]
        print(prediction)
   
        crop_image(img_tensor,pred,file.filename)
        # results = get_result(object_images)
        
    # print(results)
    # Perform additional processing using img_tensor
    # (Add your custom processing logic here)

    return JSONResponse(content={"crop":"done"}, status_code=200)