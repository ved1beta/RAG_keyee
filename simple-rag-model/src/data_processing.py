import PyPDF2
import re
import fitz
from tqdm.auto import tqdm
import spacy
import random

def text_formatter(text: str) -> str:
    """
    This function is used to format the text extracted from the pdf files.
    """
    text_clean = text.replace("\n", " ").strip()
    return text_clean

def open_read_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = []
    for page_num in tqdm(range(len(doc))):
        page = doc[page_num]
        text = page.get_text()
        text = text_formatter(text=text)

        pages_text.append({
            "page_num": page_num,
            "text": text,
            "pages_token_count": len(text)/4
        }) 
    return pages_text


nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

pages_text = open_read_pdf("data/raw/V Kishore Ayyadevara, Yeshwanth Reddy - Modern Computer Vision with PyTorch_ A Practical Roadmap Fro.pdf")

for item in tqdm(pages_text):
    doc = nlp(item["text"])
    item["sentences"] = list(doc.sents)
    item["sentence_count"] = [str(sentence) for sentence in item["sentences"]]


random_page = random.sample(pages_text, k=1)
random_page