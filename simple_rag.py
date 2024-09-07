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
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text=text)
        pages_and_texts.append({
            "page_number": page_number - 41,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count": len(text.split(". ")),
            "page_token_count": len(text) / 4,
            "text": text
        })
    return pages_and_texts

pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)

print(pages_and_texts)