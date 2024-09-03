import os
import requests
from tqdm import tqdm

pdf_path = "human-nutrition-text.pdf"

if not os.path.exists(pdf_path):
    print('Downloading...')

    url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    if response.status_code == 200:
        with open(pdf_path, 'wb') as file, tqdm(
            desc=pdf_path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                bar.update(len(data))
        print("FILE DOWNLOADED")
    else: 
        print(f"Failed to download the file: {response.status_code}")


import fitz


def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc.fitz.open(pdf_path)
    pages_and_texts = []
    