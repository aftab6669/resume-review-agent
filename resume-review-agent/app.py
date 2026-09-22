import json
import time

import streamlit as st
from groq import Groq, RateLimitError, AuthenticationError, APIError
from pypdf import PdfReader

MODEL_NAME = "openai/gpt-oss-120b"
MAX_RESUME_CHARS = 50000
MAX_JOB_CHARS = 30000

st.set_page_config(
    page_title="AI Resume Review Agent",
    page_icon="📄",
    layout="wide",
)

SYSTEM_PROMPT = """
You are a careful, evidence-based resume review agent.

Your job is to compare a candidate resume against a target job description.

NON-FABRICATION RULES:
1. Use ONLY information explicitly present in the resume.
2. Never invent degrees, certifications, skills, job titles, years of experience,
   achievements, employers, tools, languages, publications, or responsibilities.
3. If a requirement is not mentioned in the resume, classify it as "Not stated"
   rather than assuming the candidate has it.
4. Do not confuse "not stated" with "does not have".
5. Separate evidence from recommendations.
6. Recommendations must be practical and must never tell the candidate to claim
   an experience or qualification they do not actually have.
7. If the resume is ambiguous, say so.
8. Do not make hiring decisions or guarantee interview/job outcomes.

MATCHING RULE:
- Strong match: clear evidence in the resume supports the requirement.
- Partial match: related or incomplete evidence is present.
- Gap: the job requirement is clear but supporting evidence is not found.
- Not stated: the requirement cannot be assessed from the provided material.

Return ONLY valid JSON matching the requested schema.
"""

REVIEW_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "resume_review",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "overall_match": {
                    "type": "string",
                    "enum": ["Strong", "Moderate", "Limited", "Insufficient evidence"],
                },
                "summary": {"type": "string"},
                "requirement_analysis": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "requirement": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": [
                                    "Strong match",
                                    "Partial match",
                                    "Gap",
                                    "Not stated",
                                ],
                            },
                            "resume_evidence": {"type": "string"},
                            "recommendation": {"type": "string"},
                        },
                        "required": [
                            "requirement",
                            "status",
                            "resume_evidence",
                            "recommendation",
                        ],
                        "additionalProperties": False,
                    },
                },
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "priority_improvements": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "missing_or_unclear_items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "resume_quality": {
                    "type": "object",
                    "properties": {
                        "clarity": {"type": "string"},
                        "relevance": {"type": "string"},
                        "achievement_focus": {"type": "string"},
                        "formatting_advice": {"type": "string"},
                    },
                    "required": [
                        "clarity",
                        "relevance",
                        "achievement_focus",
                        "formatting_advice",
                    ],
                    "additionalProperties": False,
                },
            },
            "required": [
                "overall_match",
                "summary",
                "requirement_analysis",
                "strengths",
                "priority_improvements",
                "missing_or_unclear_items",
                "resume_quality",
            ],
            "additionalProperties": False,
        },
    },
}


def extract_pdf_text(uploaded_file):
    """Extract selectable text from a PDF."""
    try:
        reader = PdfReader(uploaded_file)
        pages = []

        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())

        combined = "\n\n".join(pages).strip()

        if not combined:
            raise ValueError(
                "No selectable text was found. This may be a scanned/image-only PDF."
            )

        return combined[:MAX_RESUME_CHARS]

    except Exception as exc:
        raise ValueError(f"PDF extraction failed: {exc}") from exc


def get_api_key():
    """Read the Groq API key from Streamlit Secrets."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None

    if not api_key or not str(api_key).strip():
        return None

    return str(api_key).strip()


def review_resume(resume_text, job_description):
    """Send the resume and job description to the single Groq agent."""
    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets."
        )

    client = Groq(api_key=api_key)

    user_prompt = f"""
Review the following candidate resume against the target job description.

CANDIDATE RESUME
----------------
{resume_text}

TARGET JOB DESCRIPTION
----------------------
{job_description}

Instructions:
- Extract the important requirements from the job description.
- Compare each requirement with explicit evidence in the resume.
- Never infer an unmentioned qualification.
- Give actionable recommendations for improving the resume.
- When recommending an improvement, phrase it conditionally if the candidate
  needs to add information that is not currently shown. Example:
  "If you have experience with X, add a specific example."
- Do not fabricate content for the candidate.
"""

    last_error = None

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=REVIEW_SCHEMA,
                temperature=0.1,
                max_tokens=5000,
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("The model returned an empty response.")

            return json.loads(content)

        except RateLimitError as exc:
            last_error = exc

            if attempt == 0:
                time.sleep(2)
                continue

            raise RuntimeError(
                "Groq rate limit reached. Please wait a little and try again."
            ) from exc

        except AuthenticationError as exc:
            raise RuntimeError(
                "Groq authentication failed. Check that GROQ_API_KEY is correct."
            ) from exc

        except APIError as exc:
            raise RuntimeError(f"Groq API error: {exc}") from exc

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "The AI returned an unexpected format. Please try again."
            ) from exc

        except Exception as exc:
            last_error = exc
            break

    raise RuntimeError(f"Review failed: {last_error}")


def show_review(review):
    st.subheader("📊 Resume Review")

    overall = review.get("overall_match", "Unknown")
    st.metric("Overall evidence-based match", overall)

    st.markdown("### Summary")
    st.write(review.get("summary", ""))

    st.markdown("### Requirement Analysis")

    requirements = review.get("requirement_analysis", [])

    if requirements:
        for item in requirements:
            status = item.get("status", "Unknown")

            with st.expander(
                f"{status}: {item.get('requirement', 'Requirement')}"
            ):
                st.markdown("**Resume evidence**")
                st.write(item.get("resume_evidence", ""))

                st.markdown("**Action**")
                st.write(item.get("recommendation", ""))
    else:
        st.info("No individual requirements were returned.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Strengths")
        for item in review.get("strengths", []):
            st.markdown(f"- {item}")

    with col2:
        st.markdown("### 🎯 Priority Improvements")
        for item in review.get("priority_improvements", []):
            st.markdown(f"- {item}")

    st.markdown("### ⚠️ Missing or Unclear Items")
    missing = review.get("missing_or_unclear_items", [])

    if missing:
        for item in missing:
            st.markdown(f"- {item}")
    else:
        st.write("No additional missing or unclear items were identified.")

    st.markdown("### ✍️ Resume Quality")
    quality = review.get("resume_quality", {})

    q1, q2 = st.columns(2)

    with q1:
        st.markdown("**Clarity**")
        st.write(quality.get("clarity", ""))

        st.markdown("**Achievement focus**")
        st.write(quality.get("achievement_focus", ""))

    with q2:
        st.markdown("**Relevance**")
        st.write(quality.get("relevance", ""))

        st.markdown("**Formatting advice**")
        st.write(quality.get("formatting_advice", ""))

    st.download_button(
        label="⬇️ Download Review as JSON",
        data=json.dumps(review, indent=2, ensure_ascii=False),
        file_name="resume_review.json",
        mime="application/json",
    )


st.title("📄 AI Resume Review Agent")
st.caption(
    "Evidence-based resume matching powered by OpenAI GPT-OSS 120B through Groq"
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
1. Paste a resume **or upload a PDF**.
2. Paste the target job description.
3. Click **Review Resume**.
4. The AI compares the two.
5. You receive evidence, gaps, and actionable improvements.

**Important:** The agent does not assume qualifications that are not stated.
"""
    )

    st.divider()
    st.caption(f"Model: `{MODEL_NAME}`")

st.subheader("1. Candidate Resume")

input_mode = st.radio(
    "Choose resume input method:",
    ["Paste resume", "Upload PDF"],
    horizontal=True,
)

resume_text = ""

if input_mode == "Paste resume":
    resume_text = st.text_area(
        "Paste the complete resume here",
        height=350,
        placeholder="Paste the candidate's resume...",
    )
else:
    uploaded_file = st.file_uploader(
        "Upload resume PDF",
        type=["pdf"],
        help="Use a text-based PDF. Scanned image-only PDFs may not contain extractable text.",
    )

    if uploaded_file is not None:
        try:
            resume_text = extract_pdf_text(uploaded_file)
            st.success("PDF text extracted successfully.")

            with st.expander("Preview extracted resume text"):
                st.text(resume_text[:5000])

        except ValueError as exc:
            st.error(str(exc))

st.subheader("2. Target Job Description")

job_description = st.text_area(
    "Paste the complete job description here",
    height=300,
    placeholder="Paste the target job description...",
)

if resume_text:
    st.caption(f"Resume text available: {len(resume_text):,} characters")

if job_description:
    st.caption(f"Job description: {len(job_description):,} characters")

st.divider()

review_button = st.button(
    "🔍 Review Resume",
    type="primary",
    use_container_width=True,
)

if review_button:
    clean_resume = resume_text.strip()
    clean_job = job_description.strip()

    if not clean_resume:
        st.error("Please paste a resume or upload a readable PDF.")

    elif not clean_job:
        st.error("Please paste the target job description.")

    elif len(clean_resume) < 100:
        st.error(
            "The resume appears too short. Please provide more complete resume content."
        )

    elif len(clean_job) < 100:
        st.error(
            "The job description appears too short. Please provide the complete job description."
        )

    else:
        with st.spinner("Analyzing the resume against the job description..."):
            try:
                result = review_resume(clean_resume, clean_job)
                st.session_state["review_result"] = result

            except RuntimeError as exc:
                st.error(str(exc))

            except Exception:
                st.error(
                    "An unexpected error occurred. Please check your inputs and try again."
                )

if "review_result" in st.session_state:
    st.divider()
    show_review(st.session_state["review_result"])

st.divider()
st.caption(
    "This tool provides resume-writing and matching assistance. It does not make hiring decisions "
    "and should not be treated as a guarantee of employment."
)
