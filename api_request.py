import requests
import os

# Specify the API endpoint URL
api_url = "http://127.0.0.1:8000/uploadfile/"

# Path to the file you want to upload
file_path = r"E:\QuickFox\OCR\OD Api\00c7ee6f-3901010008839_66779944.png"







def determine_content_type(file_path):
    # Determine content type based on file extension
    _, file_extension = os.path.splitext(file_path.lower())
    if file_extension == ".png":
        return "image/png"
    elif file_extension in {".jpg", ".jpeg"}:
        return "image/jpeg"
    else:
        raise ValueError("Unsupported file type")


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


