import requests
import os
import time
# Specify the API endpoint URL
api_url = "http://127.0.0.1:8000/getobjects/"

def determine_content_type(file_path):
    # Determine content type based on file extension
    _, file_extension = os.path.splitext(file_path.lower())
    if file_extension == ".png":
        return "image/png"
    elif file_extension in {".jpg", ".jpeg"}:
        return "image/jpeg"
    else:
        raise ValueError("Unsupported file type")

def main(file_path, api_url):
    # Open the file and create a dictionary to pass it as a payload
    with open(file_path, "rb") as file:
        content_type = determine_content_type(file_path)
        files = {"file": (file.name, file, content_type)}

    # Send a POST request to the API endpoint with the file
        response = requests.post(api_url, files=files)

    # Check the response
    if response.status_code == 200:
        print("API Response:", response.json())
    else:
        print("Error:", response.status_code, response.text)
#change to the directory containing the images
dir_path = r"E:\QuickFox\OCR\Object Detection\images"
# Loop through the files in the directory
for i, file in enumerate(os.listdir(dir_path)):
    file_path = os.path.join(dir_path, file)
    print(f"{i+1}. Processing file: {file_path}")
    main(file_path, api_url)
    # print(f"S.No: {i} File: {file_path}")
    time.sleep(5)
    # print(file_path)