# Approach Explanation (Challenge 1B)

## Overview

This project implements an intelligent PDF processing pipeline designed to extract, semantically rank, and return the most relevant document sections based on a given user profile and a task description. The primary use case is to identify important information from multi-page PDF documents with minimal manual effort. The system achieves this through a combination of layout-aware text extraction (via PyMuPDF) and semantic similarity analysis (via the `all-mpnet-base-v2` model from Sentence Transformers). The final output is a structured JSON file with metadata and a ranked list of top sections relevant to the user's intent.

---

## Problem Statement

Users often need to extract information from large PDFs such as manuals, travel brochures, academic papers, or corporate documents. However, navigating such documents manually is time-consuming. The goal is to automatically:

1. Parse multiple PDFs and extract possible sections and sub-sections.
2. Rank the sections based on how relevant they are to the user’s context.
3. Return only the most pertinent sections with refined content.

---

## Inputs

The system accepts a JSON file as input, which includes:

- **Persona**: A natural language description of the user role or background (e.g., “Travel Planner for college students”).
- **Job to be done**: A specific task or objective the user wants to accomplish (e.g., “Plan a 4-day trip for a group of 10 college students”).
- **Documents**: A list of filenames (PDFs) stored in a given directory (default: `PDFs/`).

The JSON acts as a blueprint for both document retrieval and user intent representation.

---

## Step 1: Text Extraction using PyMuPDF

The first core component is extracting structured text from the provided PDFs. The system uses `fitz`, the Python interface for PyMuPDF, to load each document and iterate through its pages.

Each page is decomposed into **blocks** and **spans**. The span objects contain metadata such as:
- Actual text
- Font name and size
- Boldness or emphasis
- Vertical positioning (y-coordinates)

Using this metadata, we classify content into:
- **Headers/Section Titles**: Bold, non-numeric strings longer than 2 characters.
- **Content**: Normal-weight, paragraph-like text elements that follow headers.

Section titles are identified by filtering spans based on their font style (e.g., bold or semibold) and content features (e.g., ignoring digits, checking length). Once headers are detected, the text in-between each header is grouped as its corresponding section content.

When no headers are detected on a page, the system defaults to treating the full page as a single content block. This ensures that important text is not skipped even if the document lacks a formal structure.

All extracted sections are stored with:
- The `title` of the section
- Cleaned `text` content (stopword removal, bullet stripping, normalization)
- `page` number of origin
- `document` filename

---

## Step 2: Cleaning and Preprocessing

To enhance semantic understanding, the extracted content undergoes cleaning:
- **Whitespace normalization**: Removes extra spaces, tabs, and line breaks.
- **Bullet point removal**: Common patterns like “•”, “-”, “*”, “1.”, “a.” are filtered out.
- **Digit cleanup**: Isolated digits at line ends are stripped.
- **Case and punctuation handling**: Uniformity in case and punctuation is maintained for better sentence encoding.

This preprocessing stage ensures that the embeddings generated later will represent the actual semantics rather than formatting artifacts.

---

## Step 3: Embedding and Semantic Ranking

This is the heart of the system where we connect the user's intent with the extracted sections.

1. **Context Creation**: The `persona` and `job_to_be_done` fields are combined into a single sentence to form the context prompt.
2. **Sentence Embedding**: The context prompt is passed through the `all-mpnet-base-v2` model from Sentence Transformers, yielding a dense vector representation (`context_embedding`).
3. **Section Embeddings**: Each extracted section (`title` + `text`) is also encoded using the same model.

To measure semantic relevance, the system computes:
- **Cosine Similarity**: A standard metric for measuring angle between two vectors. A higher value indicates greater relevance.

Sections are then **ranked** based on similarity to the user context. This ranking allows the system to surface only the most contextually important sections.

Other similarity methods (dot product, Euclidean) are also supported for experimentation, though cosine is the default.

---

## Step 4: Output Generation

The final step is to collate the top-K sections (default: 5) into a structured JSON output. The output includes:

1. **Metadata**:
   - List of processed filenames
   - Persona and job-to-be-done string
   - Timestamp of processing
2. **Extracted Sections**:
   - Document name
   - Section title
   - Page number
   - Relevance rank
3. **Subsection Analysis**:
   - Raw but cleaned section content
   - Page of origin
   - Document name

This output provides a digestible summary of what the model deems most important, giving users a high-level view and deeper content for analysis.

---

## Fault Tolerance and Logging

To ensure reliability:
- Missing or unreadable files are caught via exception handling.
- If no headers are detected in a document, fallback logic ensures no information is skipped.
- Intermediate progress is printed, helping debug issues with malformed PDFs.

---

## Extensibility and Customization

The code is modular and can be extended in the following ways:
- **OCR Support**: Tesseract or PaddleOCR can be integrated to support scanned PDFs.
- **Custom Models**: Replace `all-mpnet-base-v2` with domain-specific models like BioBERT for medical PDFs.
- **Visualization**: Top sections can be rendered in a web interface (e.g., Streamlit or React) for ease of review.

---

## Summary

This intelligent PDF ranking system effectively combines layout-aware text parsing and state-of-the-art semantic similarity to reduce information overload. Instead of forcing users to read entire documents, it surfaces only what matters most based on their needs.

Applications include:
- Travel planning
- Research report summarization
- Legal or policy document analysis
- Enterprise knowledge management

By automating the extraction and ranking process, the system improves productivity, reduces manual effort, and delivers personalized document analysis for diverse real-world use cases.
