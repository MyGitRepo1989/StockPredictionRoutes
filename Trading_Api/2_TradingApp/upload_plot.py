import requests
import dotenv
import os       

dotenv.load_dotenv()
IMGUR_CLIENT_ID = os.getenv('IMGUR_CLIENT_ID')
IMGUR_HOST_URL = 'https:image'

def hostImageImgur(image_path):
    headers = {
        'Authorization': f'Client-ID {CLIENT_ID}'
    }

    files = {
        'image': open(image_path, 'rb')
    }

    data = {
        'title': 'fileA', 
        'type': 'file',
        'description': 'A simple upload',  

    }

    response = requests.post(IMGUR_HOST_URL, headers=headers, files=files, data=data)
    
    return response.json().get('data', {}).get('link', 'No link found')

hostImageImgur('Desktop/image.jpg')