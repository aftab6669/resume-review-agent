# AI Resume Review Agent

A beginner-friendly Streamlit application that compares a candidate resume with a target job description using OpenAI GPT-OSS 120B through Groq.

## Features

- Paste resume text or upload a PDF.
- Extract selectable PDF text with pypdf.
- Compare resume evidence against job requirements.
- Never intentionally invent qualifications that are not in the resume.
- Distinguish between:
  - Strong match
  - Partial match
  - Gap
  - Not stated
- Generate practical resume-improvement recommendations.
- Download the structured review as JSON.
- Read the Groq API key from Streamlit Secrets.
- Handle missing inputs, unreadable PDFs, authentication errors, API errors, and rate limits.

## Project structure

```text
resume-review-agent/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml.example
```

## 1. Get a Groq API key

Create a Groq API key from your Groq account.

Do not put the real key inside `app.py`.

## 2. Local setup

Install Python 3.11.

Open a terminal inside this project folder and run:

```bash
python -m pip install -r requirements.txt
```

Create:

```text
.streamlit/secrets.toml
```

Put this inside it:

```toml
GROQ_API_KEY = "gsk_your_real_key_here"
```

Run:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## 3. GitHub

Upload these files:

```text
app.py
requirements.txt
README.md
.gitignore
.streamlit/secrets.toml.example
```

DO NOT upload:

```text
.streamlit/secrets.toml
```

The `.gitignore` file is already configured to exclude it.

## 4. Streamlit Community Cloud

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Choose **Create app**.
4. Select your GitHub repository.
5. Select `app.py` as the main file.
6. Open **Advanced settings**.
7. Select Python 3.11.
8. In the Secrets field, paste:

```toml
GROQ_API_KEY = "gsk_your_real_key_here"
```

9. Deploy.

Streamlit stores deployment secrets outside your Git repository.

## Model

The application uses:

```text
openai/gpt-oss-120b
```

through the Groq API.

## Important limitation

A PDF must contain selectable text. A scanned/image-only PDF may fail extraction.

The application deliberately uses an evidence-based prompt. If a qualification is not mentioned in the resume, the model should say "Not stated" rather than assuming that the candidate has it.

This is an assistance tool, not a hiring decision system.
