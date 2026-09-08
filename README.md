# HR Policy Assistant — Gradio + Groq + RAG

## Baby steps (Windows, Roman Urdu)

1. ZIP download karke Extract All karein. `hr_policy_assistant` folder kholein.
2. Python 3.11 ya 3.12 install karein (https://www.python.org/downloads/). Installation par **Add Python to PATH** tick karein. Python 3.14 is project ke liye avoid karein.
3. Folder ke address bar mein `cmd` likhein aur Enter dabayein. Neeche ke commands isi Command Prompt mein ek ek karke chalayein:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Agar Python 3.12 install ki hai toh pehle command mein `-3.11` ki jagah `-3.12` likhein. Installation internet speed ke hisaab se waqt le sakti hai; PyTorch dependency bari hai.

4. Groq API key https://console.groq.com/keys se banayein. Key chat ya GitHub mein share na karein.
5. Isi Command Prompt mein app start karein:

```bat
python app.py
```

6. Browser mein http://127.0.0.1:7860 kholein.
7. `sample_hr_policy.txt` upload karein aur **Process Policy** click karein. Pehli processing par embedding model download hoga; internet required hai. Status `Ready` hone ka wait karein.
8. App ke password box mein apni Groq API key enter karein.
9. Sawal likhein: `Permanent employees ki annual leave kitni hai?` Phir **Ask HR Assistant** click karein.
10. Sample ka expected answer: 20 working days per calendar year, prior line-manager approval ke saath. Source mein annual leave section aana chahiye.
11. Apni approved PDF/DOCX upload karein; har file change ke baad **Process Policy** dobara click karein.
12. Band karne ke liye Command Prompt mein Ctrl+C dabayein. Agli baar folder mein terminal khol kar virtual environment activate karein aur `python app.py` chalayein.

## Optional: key ko local .env file mein rakhna

Command Prompt:

```bat
copy .env.example .env
notepad .env
```

Placeholder ki jagah actual key likh kar save karein. App restart karein; phir UI mein key enter karna zaroori nahi. `.env` GitHub par upload na karein. `.gitignore` included hai. Environment variables UI mein display nahi hotay.

## Mac / Linux

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## Google Colab (optional practice)

Colab notebook mein app.py, requirements.txt aur sample_hr_policy.txt upload karein. First cell:

```python
!pip -q install -r requirements.txt
```

Second cell:

```python
from app import build_app
build_app().queue().launch(share=True, auth=("demo", "replace-with-a-strong-password"), debug=True)
```

Printed Gradio link kholein aur password se login karein. Key app ke password field mein dein. Colab runtime temporary hai; link public internet par accessible hota hai aur runtime band hone par app band hoti hai. Is route par dummy data se practice karein. Real HR deployment needs organizational access controls.

## Files

- app.py: Gradio UI, reading, chunking, embeddings, retrieval and Groq response.
- requirements.txt: required Python packages (compatible version ranges, not a fully locked environment).
- .env.example: optional local API-key configuration template.
- .gitignore: keeps local secrets/virtual environment out of Git.
- sample_hr_policy.txt: fictional test policy.
- sample_hr_policy.docx: same fictional policy in Word format.
- README.md: setup and limitations.
- TEST_REPORT.txt: checks actually performed in the creation environment.

## How it works

Files are read locally. PDF pages and DOCX paragraph/table locations are retained. Text is chunked within the multilingual embedding model's token limit with overlap. Normalized embeddings and NumPy similarity retrieve six excerpts. An optional Groq translation makes Roman Urdu questions searchable against English policies. Groq receives the question and retrieved excerpts, then returns structured JSON. The app validates source IDs and displays actual stored file/location metadata.

Default Groq model: `openai/gpt-oss-20b`. You may change GROQ_MODEL in .env to an available chat model that supports JSON object output. Model access and pricing depend on your account. The app does not call OpenAI's hosted API; the OpenAI Python package is the client for Groq.

## Practical tests

1. Annual leave: 20 working days, approval condition, correct source.
2. Sick leave: 10 working days, certificate for more than two consecutive days.
3. Overtime rate: should report missing rate, never invent a multiplier.
4. Contract staff leave: should refer to the signed contract, never assign the permanent-staff allowance.
5. Individual balance: cannot be answered from the handbook.
6. Replace files: old index and answers are cleared; processing is required again.
7. Upload scanned/empty PDF: warning or no-readable-text error, not invented content.

## Scope and limitations

This is a local learning prototype, not an HR system of record. Each question is independent; write complete questions, since there is no conversation memory. Upload/index data is session-specific and kept in memory; the index must be rebuilt after refresh/restart. Gradio temporarily stores uploaded files on the host; clearing the UI is not secure disk deletion. Only the embedding model is globally cached, not users' policies or API keys.

No OCR, payroll integration, approval submission, persistent vector database or employee login is included. DOCX headers, footers, text boxes, images and nested tables are not extracted; inspect the preview and compare key rules with the original. PDF layout and tables may extract imperfectly. Chunking can separate conditions across boundaries and retrieval can miss relevant sections. Source-ID validation checks reference validity, not factual entailment: inspect supporting text and test policies before use. Roman Urdu retrieval/translation quality varies. Missing-answer refusal is model-based and cannot guarantee zero hallucinations. Conflicting versions should be clarified with HR; the app has no authoritative policy-version registry.

Limits: 10 files, 20 MB per file, 300 pages per PDF, 5,000 chunks. These are demo limits, not protection against hostile documents. Process trusted policy files only. Requests are serialized to avoid a process/answer race in this small demo.

Groq receives the question, search translation request (if enabled), and selected policy excerpts. Local embeddings require an initial Hugging Face download. Real HR use needs approved provider/data handling, authentication, authorization, retention controls, policy ownership and evaluation. Local launch binds to 127.0.0.1 with share=False.

## Troubleshooting

- `py`/`python` not recognized: reinstall Python with Add to PATH, reopen terminal.
- Installation fails: confirm Python 3.11/3.12; activate a fresh virtual environment.
- Embedding download fails: check access to Hugging Face and your internet/proxy.
- Invalid key: replace with a Groq key, not an OpenAI key.
- Rate limit: wait and review Groq quota.
- Model unavailable: choose a currently available Groq chat model with JSON support in .env.
- Empty PDF: OCR first or use a text-based PDF/DOCX.
- Wrong answer: inspect extracted preview and retrieved sources; ask with employee type and policy version.
- Port busy: close the other app, or run `set PORT=7861` before `python app.py` in Windows Command Prompt.

Documentation checked 8 September 2026:
- https://console.groq.com/docs/models
- https://console.groq.com/docs/openai
- https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- https://www.gradio.app/docs
