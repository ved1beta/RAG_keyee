import PyPDF2
import re
import fitz
from tqdm.auto import tqdm
import spacy
import pandas as pd
import os

class PDFProcessor:
    def __init__(self, num_sentence_chunk_size=10):
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")
        self.num_sentence_chunk_size = num_sentence_chunk_size

    def text_formatter(self, text: str) -> str:
        """
        Format the text extracted from PDF files.
        """
        text_clean = text.replace("\n", " ").strip()
        return text_clean

    def open_read_pdf(self, pdf_path):
        """
        Read PDF and extract text pages.
        """
        # Validate PDF path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
            
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("File must be a PDF")
            
        try:
            doc = fitz.open(pdf_path)
            pages_text = []
            for page_num in tqdm(range(len(doc))):
                page = doc[page_num]
                text = page.get_text()
                text = self.text_formatter(text=text)
                pages_text.append({
                    "page_num": page_num,
                    "text": text,
                    "pages_token_count": len(text)/4
                }) 
            return pages_text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def split_list(self, input_list: list, slice_size: int) -> list[list[str]]:
        """
        Splits the input_list into sublists of size slice_size.
        """
        return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

    def process_pdf(self, pdf_path):
        """
        Main method to process PDF and return processed chunks.
        """
        # Read PDF pages
        pages_text = self.open_read_pdf(pdf_path)

        # Tokenize sentences
        for item in tqdm(pages_text):
            doc = self.nlp(item["text"])
            item["sentences"] = list(doc.sents)
            item["sentence_count"] = [str(sentence) for sentence in item["sentences"]]

        # Split sentences into chunks
        for item in tqdm(pages_text):
            item["sentence_chunks"] = self.split_list(
                input_list=item["sentences"],
                slice_size=self.num_sentence_chunk_size
            )
            item["num_chunks"] = len(item["sentence_chunks"])

        # Process chunks
        pages_and_chunks = []
        for item in tqdm(pages_text):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                
                # Convert Span objects to strings before joining
                joined_sentence_chunk = "".join(str(sentence) for sentence in sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) 
                chunk_dict["sentence_chunk"] = joined_sentence_chunk
                
                # Get chunk stats
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 
                
                pages_and_chunks.append(chunk_dict)

        return pd.DataFrame(pages_and_chunks)