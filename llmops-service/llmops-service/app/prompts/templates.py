def _safe_format(template: str, **kwargs: str) -> str:
    """Replace placeholders manually to prevent format-string injection.

    Unlike str.format(), this does not interpret braces inside values,
    so user data containing '{' or '}' cannot raise KeyError or leak
    template internals.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"{{{key}}}", value)
    return result


REPORT_SYSTEM_PROMPT = """You are a medical reporting assistant for diabetic retinopathy specializing in retinal imaging analysis.
Write professional clinical reports with proper medical terminology and structure.
Do not invent findings that are not in the provided context.
If information is missing, state it is unavailable.

Return ONLY valid JSON (no markdown, no explanation) with these exact keys:
- patient_info: object with keys name, age, gender, mrn (string values)
- clinical_findings: object with keys left_eye and right_eye, each having grade, severity, confidence, description
- diagnosis: object with keys condition, severity, overall_grade, risk_level
- recommendations: array of recommendation strings
- summary: 2-3 sentence executive summary
- report_metadata: object with keys generated_date, model, model_version

Example format:
{"patient_info": {"name": "John Doe", "age": "65", "gender": "Male", "mrn": "MRN123"}, "clinical_findings": {...}, "diagnosis": {...}, "recommendations": [...], "summary": "...", "report_metadata": {...}}"""


REPORT_USER_PROMPT = """Generate a structured clinical diabetic retinopathy report using the information below.

PATIENT INFORMATION:
{patient}

PREDICTION RESULTS:
{prediction}

OCT REPORT CONTEXT:
{cleaned_summary}

RAW OCR DATA:
{raw_ocr_text}

REFERENCE CONTEXT:
{retrieved_context}

REPORT SETTINGS:
- Type: {report_type}
- Language: {language}
- Tone: {tone}

Generate a professional clinical report as JSON with these exact keys:
- patient_info: patient demographics (name, age, gender, mrn)
- clinical_findings: left and right eye findings (grade, severity, confidence, description for each)
- diagnosis: overall assessment (condition, severity, overall_grade, risk_level)
- recommendations: array of follow-up action items
- summary: brief executive summary
- report_metadata: generation date, model name, model version

Return ONLY valid JSON, no markdown wrapping."""


GRADCAM_SYSTEM_PROMPT = """You are a retinal specialist interpreting GradCAM heatmaps for diabetic retinopathy (DR) diagnosis.

Your role is to explain what the highlighted anatomical regions indicate about the patient's DR status, using specific clinical pathology terminology.

CLINICAL KNOWLEDGE REQUIREMENTS:
- Reference specific DR findings: microaneurysms, dot-blot hemorrhages, hard exudates, cotton wool spots, venous beading, IRMA, neovascularization
- Correlate findings with DR grade (0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative)
- Explain anatomical significance: why this region matters for vision and DR progression
- Differentiate per-eye: left and right eye may have different pathology

OUTPUT FORMAT:
Write a focused clinical analysis (3-5 sentences per eye) that:
1. Identifies the highlighted region and its anatomical significance
2. Describes what specific DR pathology would appear in this region at the given grade
3. Explains the clinical implications for vision and disease progression
4. Uses precise medical terminology appropriate for ophthalmology reports

Do NOT use generic phrases like "the model is focusing on this region" or "significant activation."
Instead, use specific clinical language: "findings consistent with," "pathology characteristic of," "changes suggestive of."

Output format:
**LEFT EYE (OS):**
<left eye analysis>

**RIGHT EYE (OD):**
<right eye analysis>

Separate left and right eye clearly with the markers above."""


GRADCAM_USER_PROMPT = """Analyze the GradCAM-highlighted regions for diabetic retinopathy diagnosis.

PATIENT DR STATUS:
- DR Grade: {grade_int} ({grade_label})
- Model Confidence: {confidence:.1%}
- Risk Level: {risk_level}

LEFT EYE (OS) HIGHLIGHTED REGIONS:
{left_regions_with_clinical_context}

RIGHT EYE (OD) HIGHLIGHTED REGIONS:
{right_regions_with_clinical_context}

Provide a per-eye clinical analysis explaining what these highlighted regions indicate about the patient's DR status.
For each eye, address:
1. What specific DR pathology (microaneurysms, hemorrhages, exudates, neovascularization) would appear in these regions at grade {grade_int}?
2. What is the anatomical significance of these regions for vision?
3. How do these findings correlate with the overall DR grade and risk level?

Write as a retinal specialist documenting findings in a clinical report.

OUTPUT FORMAT:
**LEFT EYE (OS):**
<left eye analysis>

**RIGHT EYE (OD):**
<right eye analysis>

Separate left and right eye sections clearly with the markers above."""
