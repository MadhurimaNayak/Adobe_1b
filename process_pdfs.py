import json
import os
import fitz  # PyMuPDF
import datetime
import re
import time
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("all-mpnet-base-v2")

def flatten_text(obj):
    if isinstance(obj, dict):
        return " ".join(str(v) for v in obj.values())
    elif isinstance(obj, list):
        return " ".join(map(str, obj))
    else:
        return str(obj)

def is_bullet_point(text):
    bullet_patterns = [
        r'^\u2022\s+', r'^-\s+', r'^\*\s+', r'^\d+\.\s+',
        r'^[a-zA-Z]\.\s+', r'^[ivxlcdm]+\.\s+', r'^\([a-zA-Z0-9]+\)\s+',
    ]
    return any(re.match(pattern, text, re.IGNORECASE) for pattern in bullet_patterns)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b\d+\b(?=\s*$)', '', text)
    text = re.sub(r'^[^\w]*', '', text)
    return text.strip()

def extract_sections_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    for page_number, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        text_elements = []
        for block in blocks:
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            text_elements.append({
                                "text": text,
                                "font": span["font"].lower(),
                                "y": span["bbox"][1],
                                "is_bold": "bold" in span["font"].lower(),
                                "size": span.get("size", 0)
                            })
        text_elements.sort(key=lambda x: x["y"])
        headers = []
        for i, el in enumerate(text_elements):
            if el["is_bold"] and len(el["text"]) > 2 and not el["text"].isdigit() and not re.match(r'^\d+\.\d*$', el["text"]):
                headers.append({"title": el["text"], "index": i, "y": el["y"]})
        for i, header in enumerate(headers):
            start = header["index"] + 1
            end = headers[i + 1]["index"] if i + 1 < len(headers) else len(text_elements)
            content = [el["text"] for el in text_elements[start:end] if not el["is_bold"] and len(el["text"].strip()) > 1]
            content = clean_text(" ".join(content))
            if content and len(content) > 10:
                sections.append({
                    "title": header["title"],
                    "text": content,
                    "page": page_number,
                    "document": os.path.basename(pdf_path)
                })
        if not headers and text_elements:
            content = clean_text(" ".join([el["text"] for el in text_elements if not el["is_bold"]]))
            if content and len(content) > 10:
                sections.append({
                    "title": f"Page {page_number} Content",
                    "text": content,
                    "page": page_number,
                    "document": os.path.basename(pdf_path)
                })
    doc.close()
    return sections

def rank_sections(context_embedding, sections, method="cosine"):
    if not sections:
        return []
    texts = [f"{s['title']} {s['text']}" for s in sections]
    embeddings = model.encode(texts, convert_to_tensor=True)
    if method == "cosine":
        scores = util.cos_sim(context_embedding, embeddings)[0]
    elif method == "dot":
        scores = torch.matmul(context_embedding, embeddings.T)[0]
    elif method == "euclidean":
        from torch.nn.functional import pairwise_distance
        scores = -pairwise_distance(context_embedding.unsqueeze(0), embeddings).squeeze(0)
    else:
        raise ValueError(f"Invalid ranking method: {method}")
    ranked = sorted(zip(sections, scores), key=lambda x: x[1], reverse=True)
    for rank, (section, score) in enumerate(ranked, 1):
        section['importance_rank'] = rank
        section['relevance_score'] = float(score)
    return [s for s, _ in ranked]

def process_documents(json_path, base_path="PDFs", top_k=5, rank_method="cosine"):
    with open(json_path, "r", encoding='utf-8') as f:
        input_data = json.load(f)
    documents = input_data.get("documents", [])
    persona = flatten_text(input_data.get("persona", ""))
    job = flatten_text(input_data.get("job_to_be_done", ""))
    context = f"{persona} {job}"
    context_embedding = model.encode(context, convert_to_tensor=True)
    metadata = {
        "input_documents": [doc.get("filename", "unknown") for doc in documents],
        "persona": persona,
        "job_to_be_done": job,
        "processing_timestamp": datetime.datetime.now().isoformat()
    }
    all_sections = []
    for doc_info in documents:
        filename = doc_info.get("filename", "")
        full_path = os.path.join(base_path, filename)
        if os.path.exists(full_path):
            try:
                sections = extract_sections_from_pdf(full_path)
                all_sections.extend(sections)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    #metadata["total_sections_extracted"] = len(all_sections)
    if not all_sections:
        return {"metadata": metadata, "extracted_sections": [], "subsection_analysis": []}
    ranked = rank_sections(context_embedding, all_sections, method=rank_method)[:top_k]
    extracted_sections = [
        {
            "document": s["document"],
            "section_title": s["title"],
            "importance_rank": s["importance_rank"],
            "page_number": s["page"]
        } for s in ranked
    ]
    subsection_analysis = [
        {
            "document": s["document"],
            "refined_text": s["text"],
            "page_number": s["page"]
        } for s in ranked
    ]
    return {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

def main():
    output = process_documents(
        json_path="challenge1b_input.json",
        base_path="PDFs",
        top_k=5,
        rank_method="cosine"
    )
    if output:
        with open("challenge1b_output.json", "w", encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        print("Output saved to challenge1b_output.json")

if __name__ == "__main__":
    main()
