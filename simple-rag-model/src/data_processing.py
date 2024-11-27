import PyPDF2
import re
import fitz
from tqdm.auto import tqdm
import spacy
import random
import pandas as pd

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

pages_text = open_read_pdf("data.pdf")

for item in tqdm(pages_text):
    doc = nlp(item["text"])
    item["sentences"] = list(doc.sents)
    item["sentence_count"] = [str(sentence) for sentence in item["sentences"]]


random_page = random.sample(pages_text, k=1)

num_sentence_chunk_size = 10 

# Create a function that recursively splits a list into desired sizes
def split_list(input_list: list, 
               slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

# Loop through pages and texts and split sentences into chunks
for item in tqdm(pages_text):
    item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                         slice_size=num_sentence_chunk_size)
    item["num_chunks"] = len(item["sentence_chunks"])


# Split each chunk into its own item
pages_and_chunks = []
for item in tqdm(pages_text):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        
        
        # Convert Span objects to strings before joining
        joined_sentence_chunk = "".join(str(sentence) for sentence in sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo 
        chunk_dict["sentence_chunk"] = joined_sentence_chunk
        # Get stats about the chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters
        
        pages_and_chunks.append(chunk_dict)

# How many chunks do we have?
print(len(pages_and_chunks))

df = pd.DataFrame(pages_and_chunks) 
