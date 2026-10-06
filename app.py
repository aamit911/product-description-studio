
import json
import os
import random
import re
import time
from io import BytesIO

import pandas as pd
import streamlit as st
from ollama import chat as ollama_chat
from google import genai
from google.genai import types

# ============================================================
# Configuration
# ============================================================
LOCAL_MODEL = "qwen3:4b"
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
]
GEMINI_MAX_RETRIES = 2
GEMINI_INITIAL_BACKOFF_SECONDS = 1.5

REQUIRED_COLUMNS = [
    "Product Name",
    "Category",
    "Brand",
    "Material",
    "Target Customer",
    "Features",
]

st.set_page_config(
    page_title="AI Product Catalog Enrichment Studio",
    page_icon="📦",
    layout="wide",
)


# ============================================================
# Helpers
# ============================================================
def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [str(value)]


def get_gemini_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_gemini_client():
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Set GEMINI_API_KEY in Streamlit secrets "
            "or as a Windows environment variable. Do not put the key in app.py."
        )
    return genai.Client(api_key=api_key)


# ============================================================
# MVP 1.1 — Brand Profile
# ============================================================
def default_brand_profile():
    return {
        "brand_name": "",
        "tone": "Professional, clear and natural",
        "writing_style": "Clear, concise and customer-friendly",
        "target_audience": "",
        "preferred_terms": "",
        "forbidden_terms": "",
        "max_title_chars": 0,
        "max_description_chars": 0,
    }


def init_brand_profile():
    if "brand_profile" not in st.session_state:
        st.session_state["brand_profile"] = default_brand_profile()


def render_brand_profile():
    st.sidebar.divider()
    st.sidebar.header("🏷️ Brand Profile")
    st.sidebar.caption(
        "Controls style and presentation only. Brand instructions are never treated "
        "as product evidence and cannot justify unsupported product claims."
    )

    profile = st.session_state["brand_profile"]

    profile["brand_name"] = st.sidebar.text_input(
        "Brand Name",
        value=profile.get("brand_name", ""),
        key="brand_profile_name",
    )
    profile["tone"] = st.sidebar.text_input(
        "Tone",
        value=profile.get("tone", ""),
        key="brand_profile_tone",
    )
    profile["writing_style"] = st.sidebar.text_area(
        "Writing Style",
        value=profile.get("writing_style", ""),
        height=80,
        key="brand_profile_style",
    )
    profile["target_audience"] = st.sidebar.text_input(
        "Target Audience",
        value=profile.get("target_audience", ""),
        key="brand_profile_audience",
    )
    profile["preferred_terms"] = st.sidebar.text_input(
        "Preferred Terms",
        value=profile.get("preferred_terms", ""),
        key="brand_profile_preferred",
        help="Comma-separated terms.",
    )
    profile["forbidden_terms"] = st.sidebar.text_input(
        "Forbidden Terms",
        value=profile.get("forbidden_terms", ""),
        key="brand_profile_forbidden",
        help="Comma-separated terms.",
    )
    profile["max_title_chars"] = int(
        st.sidebar.number_input(
            "Maximum Product Title Characters",
            min_value=0,
            max_value=500,
            value=int(profile.get("max_title_chars", 0) or 0),
            step=1,
            key="brand_profile_title_limit",
            help="0 means no explicit limit.",
        )
    )
    profile["max_description_chars"] = int(
        st.sidebar.number_input(
            "Maximum Description Characters",
            min_value=0,
            max_value=5000,
            value=int(profile.get("max_description_chars", 0) or 0),
            step=1,
            key="brand_profile_description_limit",
            help="0 means no explicit limit.",
        )
    )

    st.sidebar.success("Brand profile active")
    return profile


def build_brand_instructions(profile):
    if not profile:
        return "No brand profile was supplied. Use neutral professional ecommerce language."

    def clean_terms(value):
        return [x.strip() for x in str(value or "").split(",") if x.strip()]

    preferred = clean_terms(profile.get("preferred_terms"))
    forbidden = clean_terms(profile.get("forbidden_terms"))

    title_limit = int(profile.get("max_title_chars") or 0)
    description_limit = int(profile.get("max_description_chars") or 0)

    return f"""
BRAND PROFILE — STYLE ONLY
These instructions control wording, tone and presentation. They are NOT product
evidence and must NEVER be used to invent or strengthen a product fact.

Brand name: {profile.get("brand_name", "")}
Tone: {profile.get("tone", "")}
Writing style: {profile.get("writing_style", "")}
Target audience: {profile.get("target_audience", "")}
Preferred terms: {", ".join(preferred) if preferred else "None specified"}
Forbidden terms: {", ".join(forbidden) if forbidden else "None specified"}
Maximum product title characters: {title_limit if title_limit > 0 else "No explicit limit"}
Maximum description characters: {description_limit if description_limit > 0 else "No explicit limit"}

Apply these instructions only when they do not conflict with source-grounding rules.
Do not add a preferred term if doing so creates an unsupported factual claim.
Do not use forbidden terms.
"""


# ============================================================
# AI prompts and schemas
# ============================================================
def content_schema():
    evidence_item = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "source_evidence": {"type": "string"},
        },
        "required": ["text", "source_evidence"],
    }

    claim_item = {
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "source_evidence": {"type": "string"},
        },
        "required": ["claim", "source_evidence"],
    }

    return {
        "type": "object",
        "properties": {
            "product_title": {"type": "string"},
            "short_description": {"type": "string"},
            "long_description": {"type": "string"},
            "key_features": {"type": "array", "items": evidence_item},
            "product_benefits": {"type": "array", "items": evidence_item},
            "seo_title": {"type": "string"},
            "seo_meta_description": {"type": "string"},
            "seo_keywords": {"type": "array", "items": {"type": "string"}},
            "claims": {"type": "array", "items": claim_item},
        },
        "required": [
            "product_title",
            "short_description",
            "long_description",
            "key_features",
            "product_benefits",
            "seo_title",
            "seo_meta_description",
            "seo_keywords",
            "claims",
        ],
    }


def audit_schema():
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["PASS", "FAIL"]},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "reason": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                    },
                    "required": ["claim", "reason", "severity"],
                },
            },
        },
        "required": ["status", "issues"],
    }


def build_product_prompt(product, brand_profile):
    return f"""
You are an enterprise Product Information Management (PIM) content specialist.

Create ecommerce product content from ONLY the supplied product data.

CRITICAL FACTUALITY AND PROVENANCE RULES:
1. Never invent product facts.
2. Never add specifications, dimensions, weights, percentages, certifications,
   warranties, compatibility, ingredients, performance claims, safety claims,
   origin, standards, or materials unless explicitly supplied.
3. Do not strengthen a claim. "lightweight" must not become "ultra-lightweight";
   "cotton" must not become "100% cotton".
4. Do not infer facts from product category or common knowledge.
5. Preserve numbers and units exactly.
6. Every factual claim in key_features, product_benefits and claims MUST include
   source_evidence copied VERBATIM from the supplied product data.
7. If a claim cannot be supported by exact source evidence, do not make it.
8. Marketing wording is allowed only around supported facts and must not introduce
   a new factual claim.
9. Brand-profile instructions below control STYLE ONLY. They are not evidence.
10. Never use a brand instruction to create a factual product claim.
11. Respect forbidden terms and requested length limits where possible without
    violating factuality.
12. Keep the generated content natural and useful for ecommerce.

BRAND PROFILE:
{build_brand_instructions(brand_profile)}

SUPPLIED PRODUCT DATA:
Product Name: {product.get("Product Name", "")}
Category: {product.get("Category", "")}
Brand: {product.get("Brand", "")}
Material: {product.get("Material", "")}
Target Customer: {product.get("Target Customer", "")}
Features: {product.get("Features", "")}

Return JSON matching the required schema.
"""


def build_audit_prompt(product, output, brand_profile):
    return f"""
You are a strict PIM content quality auditor.

Compare SOURCE DATA against GENERATED CONTENT.

BRAND PROFILE:
{build_brand_instructions(brand_profile)}

Important: brand style instructions are NOT product evidence.

SOURCE DATA:
{json.dumps(product, ensure_ascii=False, indent=2)}

GENERATED CONTENT:
{json.dumps(output, ensure_ascii=False, indent=2)}

Identify only factual claims in generated content that are:
- unsupported by the source
- strengthened beyond the source
- contradicted by the source
- numerically inconsistent with the source

Examples:
- Source says "cotton"; generated says "100% cotton".
- Source says "lightweight"; generated says "ultra-lightweight".
- Source has no waterproof claim; generated says "waterproof".
- Source says EVA sole; generated says "shock-absorbing EVA sole" without source support.

Return PASS only if there are no meaningful factual overclaims.
"""


# ============================================================
# Generation
# ============================================================
def parse_json(raw):
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r"^```json\s*", "", str(raw).strip(), flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)


def is_transient_gemini_error(exc):
    code = getattr(exc, "code", None)
    if code in {408, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in [
            "503", "unavailable", "service unavailable", "high demand",
            "temporarily overloaded", "429", "resource exhausted",
            "500", "502", "504", "deadline exceeded", "timeout",
        ]
    )


def gemini_error_message(exc):
    code = getattr(exc, "code", None)
    return f"HTTP {code}: {exc}" if code else str(exc)


def generate_with_gemini(client, prompt, schema):
    failures = []
    for model_index, model in enumerate(GEMINI_MODELS):
        for attempt in range(GEMINI_MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                st.session_state["last_gemini_model"] = model
                st.session_state["last_gemini_fallback"] = model_index > 0
                return parse_json(response.text)
            except Exception as exc:
                if not is_transient_gemini_error(exc):
                    raise
                failures.append(
                    f"{model} attempt {attempt + 1}: {gemini_error_message(exc)}"
                )
                if attempt < GEMINI_MAX_RETRIES:
                    delay = GEMINI_INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    delay += random.uniform(0, 0.5)
                    time.sleep(delay)

        if model_index < len(GEMINI_MODELS) - 1:
            st.warning(
                f"Gemini model `{model}` is temporarily unavailable. "
                f"Trying `{GEMINI_MODELS[model_index + 1]}`..."
            )

    raise RuntimeError(
        "Gemini could not complete the request after retries and fallback models. "
        + " | ".join(failures[-6:])
    )


def generate_json(provider, prompt, schema, temperature=0.1):
    if provider == "Local — Ollama / Qwen3 4B":
        response = ollama_chat(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": temperature},
        )
        return parse_json(response["message"]["content"])

    if provider == "Cloud — Gemini":
        return generate_with_gemini(get_gemini_client(), prompt, schema)

    raise ValueError(f"Unsupported AI engine: {provider}")


def generate_content(product, provider, brand_profile):
    return generate_json(
        provider,
        build_product_prompt(product, brand_profile),
        content_schema(),
        temperature=0.1,
    )


# ============================================================
# Validation
# ============================================================
def source_text(product):
    return " ".join(
        clean_value(v) for v in product.values() if clean_value(v)
    )


def validate_provenance(product, output):
    source = source_text(product)
    problems = []
    checked = 0

    def check_claim(item, location):
        nonlocal checked
        if not isinstance(item, dict):
            problems.append(f"{location}: claim is not a structured object.")
            return

        claim = clean_value(item.get("claim") or item.get("text"))
        evidence = clean_value(item.get("source_evidence"))

        if not claim:
            problems.append(f"{location}: missing claim/text.")
            return

        checked += 1

        if not evidence:
            problems.append(f"{location}: missing source evidence.")
        elif evidence not in source:
            problems.append(
                f"{location}: source evidence was not found verbatim in the "
                f"supplied product data: {evidence}"
            )

    for i, item in enumerate(output.get("claims", [])):
        check_claim(item, f"Claim {i + 1}")
    for i, item in enumerate(output.get("key_features", [])):
        check_claim(item, f"Key feature {i + 1}")
    for i, item in enumerate(output.get("product_benefits", [])):
        check_claim(item, f"Benefit {i + 1}")

    return problems, checked


def validate_numbers(input_product, output):
    source = source_text(input_product)
    output_text = json.dumps(output, ensure_ascii=False)
    input_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", source))
    output_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", output_text))
    return sorted(output_numbers - input_numbers)


def validate_required_evidence(output):
    problems = []
    for field in ["key_features", "product_benefits"]:
        items = output.get(field, [])
        if not isinstance(items, list):
            problems.append(f"{field} is not a list.")
            continue
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                problems.append(f"{field} item {i + 1} has no provenance object.")
    return problems


def validate_brand_terms(output, brand_profile):
    problems = []
    forbidden = [
        x.strip().lower()
        for x in str(brand_profile.get("forbidden_terms", "")).split(",")
        if x.strip()
    ]

    if not forbidden:
        return problems

    output_text = json.dumps(output, ensure_ascii=False).lower()

    for term in forbidden:
        if term in output_text:
            problems.append(
                f"Brand profile: forbidden term used in generated content: '{term}'"
            )

    return problems


def validate_length_limits(output, brand_profile):
    problems = []
    title_limit = int(brand_profile.get("max_title_chars") or 0)
    desc_limit = int(brand_profile.get("max_description_chars") or 0)

    title = clean_value(output.get("product_title"))
    short_desc = clean_value(output.get("short_description"))
    long_desc = clean_value(output.get("long_description"))

    if title_limit and len(title) > title_limit:
        problems.append(
            f"Brand profile: product title is {len(title)} characters; "
            f"limit is {title_limit}."
        )

    if desc_limit:
        for name, value in [
            ("short description", short_desc),
            ("long description", long_desc),
        ]:
            if len(value) > desc_limit:
                problems.append(
                    f"Brand profile: {name} is {len(value)} characters; "
                    f"limit is {desc_limit}."
                )

    return problems


def audit_content_with_model(product, output, provider, brand_profile):
    return generate_json(
        provider,
        build_audit_prompt(product, output, brand_profile),
        audit_schema(),
        temperature=0.0,
    )


def validate_content(product, output, provider, brand_profile, run_ai_audit=True):
    problems, checked = validate_provenance(product, output)
    problems.extend(validate_required_evidence(output))
    problems.extend(validate_brand_terms(output, brand_profile))
    problems.extend(validate_length_limits(output, brand_profile))

    unexpected_numbers = validate_numbers(product, output)
    if unexpected_numbers:
        problems.append(
            "Potential unsupported numeric values detected: "
            + ", ".join(unexpected_numbers)
        )

    audit = None
    if run_ai_audit:
        try:
            audit = audit_content_with_model(
                product, output, provider, brand_profile
            )
            if audit.get("status") == "FAIL":
                for issue in audit.get("issues", []):
                    problems.append(
                        "AI audit: "
                        + clean_value(issue.get("claim"))
                        + " — "
                        + clean_value(issue.get("reason"))
                    )
        except Exception as exc:
            problems.append(f"AI audit could not be completed: {exc}")

    return problems, checked, audit


# ============================================================
# Export / review helpers
# ============================================================
def flatten_provenance(items):
    values = []
    for item in normalize_list(items):
        if isinstance(item, dict):
            text_value = clean_value(item.get("text") or item.get("claim"))
            evidence = clean_value(item.get("source_evidence"))
            if text_value:
                values.append(text_value)
            if evidence:
                values.append(f"[Evidence: {evidence}]")
        else:
            values.append(str(item))
    return " | ".join(values)


def content_to_row(product, result, validation_problems=None, checked_claims=0,
                   audit=None, provider="", brand_profile=None):
    validation_problems = validation_problems or []
    return {
        "Product Name": product.get("Product Name", ""),
        "Category": product.get("Category", ""),
        "Brand": product.get("Brand", ""),
        "Material": product.get("Material", ""),
        "Target Customer": product.get("Target Customer", ""),
        "Features": product.get("Features", ""),
        "Product Title": result.get("product_title", ""),
        "Short Description": result.get("short_description", ""),
        "Long Description": result.get("long_description", ""),
        "Key Features": flatten_provenance(result.get("key_features")),
        "Product Benefits": flatten_provenance(result.get("product_benefits")),
        "SEO Title": result.get("seo_title", ""),
        "SEO Meta Description": result.get("seo_meta_description", ""),
        "SEO Keywords": " | ".join(normalize_list(result.get("seo_keywords"))),
        "Validation Status": "PASS" if not validation_problems else "REVIEW",
        "Evidence Checks": checked_claims,
        "Validation Issues": " | ".join(validation_problems),
        "AI Audit": (
            audit.get("status", "UNKNOWN")
            if isinstance(audit, dict) else "UNKNOWN"
        ),
        "AI Engine": provider,
        "Brand Profile": (
            brand_profile.get("brand_name", "")
            if isinstance(brand_profile, dict) else ""
        ),
        "Status": "Generated" if not validation_problems else "Generated - Review",
    }


def dataframe_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Generated Content")
    output.seek(0)
    return output


def init_review_state():
    defaults = {
        "single_product": None,
        "single_result": None,
        "single_validation_problems": [],
        "single_checked_claims": 0,
        "single_audit": None,
        "single_approved": False,
        "bulk_result_df": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_evidence_items(title, items):
    st.markdown(f"### {title}")
    normalized = normalize_list(items)

    if not normalized:
        st.caption("No items generated.")
        return

    for idx, item in enumerate(normalized, start=1):
        if isinstance(item, dict):
            text_value = clean_value(item.get("text") or item.get("claim"))
            evidence = clean_value(item.get("source_evidence"))
            st.markdown(f"**{idx}.** {text_value}")
            if evidence:
                st.caption(f"Evidence: {evidence}")
        else:
            st.markdown(f"**{idx}.** {item}")


def render_single_review(provider):
    product = st.session_state.get("single_product")
    result = st.session_state.get("single_result")

    if not product or not result:
        return

    st.divider()
    st.subheader("🔎 Product Review")
    st.caption(
        "Review generated content, evidence and validation before approving."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Source Product Data")
        for key, value in product.items():
            st.markdown(f"**{key}:** {value or '—'}")

    with right:
        st.markdown("### Generated Content")
        st.markdown(f"**Product Title:** {result.get('product_title', '')}")
        st.markdown(f"**Short Description:** {result.get('short_description', '')}")
        st.markdown(f"**Long Description:** {result.get('long_description', '')}")
        st.markdown(f"**SEO Title:** {result.get('seo_title', '')}")
        st.markdown(
            f"**SEO Meta Description:** "
            f"{result.get('seo_meta_description', '')}"
        )
        st.markdown(
            f"**SEO Keywords:** "
            f"{', '.join(normalize_list(result.get('seo_keywords')))}"
        )

    a, b = st.columns(2)
    with a:
        render_evidence_items("Key Features", result.get("key_features"))
    with b:
        render_evidence_items("Product Benefits", result.get("product_benefits"))

    problems = st.session_state.get("single_validation_problems", [])
    checked = st.session_state.get("single_checked_claims", 0)
    audit = st.session_state.get("single_audit")

    st.markdown("### Validation & Audit")

    if problems:
        st.warning(f"REVIEW REQUIRED — {len(problems)} issue(s) detected.")
        for issue in problems:
            st.write(f"• {issue}")
    else:
        st.success(f"PASS — {checked} factual item(s) have source evidence.")

    if audit:
        if audit.get("status") == "PASS":
            st.success("AI factuality audit: PASS")
        else:
            st.warning("AI factuality audit: FAIL / REVIEW")
            for issue in audit.get("issues", []):
                st.write(
                    f"• {clean_value(issue.get('claim'))}: "
                    f"{clean_value(issue.get('reason'))}"
                )

    st.markdown("### Approval")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("✏️ Regenerate", key="single_regenerate"):
            st.session_state["single_result"] = None
            st.session_state["single_approved"] = False
            st.rerun()

    with c2:
        if st.button("🔄 Re-run Validation", key="single_revalidate"):
            try:
                vp, cc, au = validate_content(
                    product,
                    result,
                    provider,
                    st.session_state["brand_profile"],
                    run_ai_audit=True,
                )
                st.session_state["single_validation_problems"] = vp
                st.session_state["single_checked_claims"] = cc
                st.session_state["single_audit"] = au
                st.session_state["single_approved"] = False
                st.rerun()
            except Exception as exc:
                st.error(f"Validation failed: {exc}")

    with c3:
        if st.button(
            "✅ Approve Product",
            key="single_approve",
            type="primary",
            disabled=bool(problems),
        ):
            st.session_state["single_approved"] = True
            st.success("Product approved for export.")

    if st.session_state.get("single_approved"):
        approved_df = pd.DataFrame(
            [
                content_to_row(
                    product,
                    result,
                    problems,
                    checked,
                    audit,
                    provider,
                    st.session_state["brand_profile"],
                )
            ]
        )
        st.download_button(
            "⬇️ Export Approved Product",
            data=dataframe_to_excel(approved_df),
            file_name="approved_product.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="single_export_approved",
        )


def render_bulk_summary(result_df):
    total = len(result_df)
    passed = int((result_df["Validation Status"] == "PASS").sum())
    review = int((result_df["Validation Status"] == "REVIEW").sum())
    failed = int((result_df["Validation Status"] == "FAILED").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total)
    c2.metric("Passed", passed)
    c3.metric("Review", review)
    c4.metric("Failed", failed)


def render_bulk_review_table(result_df):
    st.subheader("🔎 Bulk Review Queue")

    review_df = result_df[
        result_df["Validation Status"].isin(["REVIEW", "FAILED"])
    ].copy()

    if review_df.empty:
        st.success("All generated products passed validation.")
        return

    st.warning(f"{len(review_df)} product(s) require review.")

    display_columns = [
        "Product Name",
        "Product Title",
        "Validation Status",
        "Evidence Checks",
        "AI Audit",
        "Validation Issues",
    ]

    st.dataframe(
        review_df[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    selected = st.selectbox(
        "Select a product to inspect",
        review_df["Product Name"].tolist(),
        key="bulk_review_product",
    )

    selected_row = review_df[
        review_df["Product Name"] == selected
    ].iloc[0]

    with st.expander("View source and generated content", expanded=True):
        left, right = st.columns(2)

        with left:
            st.markdown("**Source data**")
            for col in [
                "Category",
                "Brand",
                "Material",
                "Target Customer",
                "Features",
            ]:
                st.markdown(f"**{col}:** {selected_row.get(col, '')}")

        with right:
            st.markdown("**Generated content**")
            st.markdown(f"**Title:** {selected_row.get('Product Title', '')}")
            st.markdown(
                f"**Short description:** "
                f"{selected_row.get('Short Description', '')}"
            )
            st.markdown(
                f"**Long description:** "
                f"{selected_row.get('Long Description', '')}"
            )
            st.markdown(
                f"**SEO title:** {selected_row.get('SEO Title', '')}"
            )
            st.markdown(
                f"**SEO description:** "
                f"{selected_row.get('SEO Meta Description', '')}"
            )

        st.markdown("**Validation issues**")
        st.write(selected_row.get("Validation Issues", "None"))


# ============================================================
# Application
# ============================================================
init_brand_profile()
init_review_state()

st.title("📦 AI Product Catalog Enrichment Studio")
st.caption(
    "MVP 1.1 — Generate → Validate → Review → Approve → Export | "
    "Brand Profile included"
)

with st.sidebar:
    st.header("⚙️ AI Engine")

    provider = st.radio(
        "Choose inference engine",
        [
            "Local — Ollama / Qwen3 4B",
            "Cloud — Gemini",
        ],
        index=0,
    )

    if provider.startswith("Cloud"):
        if get_gemini_api_key():
            st.success("Gemini API key detected")
        else:
            st.warning("Gemini API key not detected")
            st.caption("Set GEMINI_API_KEY before using Cloud mode.")

        st.caption(
            "Primary: gemini-3.8-flash | "
            "Fallbacks: gemini-3.7-flash → gemini-3.6-flash"
        )
    else:
        st.success("Local mode selected")
        st.caption("Model: qwen3:4b")

    st.divider()
    st.caption(
        "Provenance and factuality validation run after generation regardless "
        "of the selected model."
    )

brand_profile = render_brand_profile()

if brand_profile.get("brand_name"):
    st.info(
        f"Active brand profile: **{brand_profile['brand_name']}**. "
        "Style rules will affect generation, but product facts remain source-grounded."
    )

tab_single, tab_bulk = st.tabs(["Single Product", "Bulk Excel"])


# ============================================================
# Single Product
# ============================================================
with tab_single:
    st.subheader("Generate content for one product")

    left, right = st.columns(2)

    with left:
        product_name = st.text_input("Product Name", key="single_name")
        category = st.text_input("Category", key="single_category")
        brand = st.text_input("Brand", key="single_brand")
        target_customer = st.text_input(
            "Target Customer",
            key="single_customer",
        )

    with right:
        material = st.text_input("Material", key="single_material")
        features = st.text_area(
            "Product Features",
            height=140,
            placeholder=(
                "Example: Lightweight, breathable mesh, EVA cushioning, "
                "non-slip outsole"
            ),
            key="single_features",
        )

    generate_single = st.button(
        "🚀 Generate Product Content",
        type="primary",
        key="generate_single",
    )

    if generate_single:
        if not product_name:
            st.error("Product Name is required.")
        elif not features and not material:
            st.error("Provide at least some product attributes or features.")
        else:
            product = {
                "Product Name": product_name,
                "Category": category,
                "Brand": brand,
                "Material": material,
                "Target Customer": target_customer,
                "Features": features,
            }

            engine_name = (
                "Gemini" if provider.startswith("Cloud") else "Qwen3 4B"
            )

            with st.spinner(f"Generating with {engine_name}..."):
                try:
                    result = generate_content(
                        product,
                        provider,
                        brand_profile,
                    )

                    with st.spinner(
                        "Running provenance, brand and factuality checks..."
                    ):
                        validation_problems, checked_claims, audit = (
                            validate_content(
                                product,
                                result,
                                provider,
                                brand_profile,
                                run_ai_audit=True,
                            )
                        )

                    st.session_state["single_product"] = product
                    st.session_state["single_result"] = result
                    st.session_state["single_validation_problems"] = (
                        validation_problems
                    )
                    st.session_state["single_checked_claims"] = checked_claims
                    st.session_state["single_audit"] = audit
                    st.session_state["single_approved"] = False

                    st.success(
                        "Content generated and validation completed. "
                        "Review the product below."
                    )
                    st.rerun()

                except Exception as exc:
                    st.error(f"Generation failed: {exc}")

    render_single_review(provider)


# ============================================================
# Bulk Excel
# ============================================================
with tab_bulk:
    st.subheader("Generate content for multiple products")

    st.write(
        "Upload an Excel file. The first row must contain these columns: "
        "**Product Name, Category, Brand, Material, Target Customer, Features**."
    )

    template = pd.DataFrame(
        [
            {
                "Product Name": "Men's Running Shoes",
                "Category": "Sports Footwear",
                "Brand": "TestBrand",
                "Material": "Mesh upper, EVA sole",
                "Target Customer": "Adults who run regularly",
                "Features": (
                    "Lightweight, breathable mesh, EVA cushioning, "
                    "non-slip outsole"
                ),
            },
            {
                "Product Name": "Cotton Casual Shirt",
                "Category": "Men's Apparel",
                "Brand": "TestBrand",
                "Material": "Cotton",
                "Target Customer": "Adults",
                "Features": "Full sleeves, button closure, casual fit",
            },
        ]
    )

    st.download_button(
        "⬇️ Download Excel Template",
        data=dataframe_to_excel(template),
        file_name="product_content_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    uploaded_file = st.file_uploader(
        "Upload product Excel file",
        type=["xlsx"],
    )

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)

            missing = [
                column for column in REQUIRED_COLUMNS
                if column not in df.columns
            ]

            if missing:
                st.error(
                    "Missing required columns: " + ", ".join(missing)
                )
            else:
                st.success(f"Loaded {len(df)} products.")
                st.dataframe(
                    df.head(10),
                    use_container_width=True,
                )

                generate_bulk = st.button(
                    "🚀 Generate Content for All Products",
                    type="primary",
                    key="generate_bulk",
                )

                if generate_bulk:
                    results = []
                    progress = st.progress(0)
                    status = st.empty()

                    for index, row in df.iterrows():
                        product = {
                            column: clean_value(row[column])
                            for column in REQUIRED_COLUMNS
                        }

                        status.write(
                            f"Processing product {index + 1} of "
                            f"{len(df)}: {product['Product Name']}"
                        )

                        try:
                            generated = generate_content(
                                product,
                                provider,
                                brand_profile,
                            )

                            validation_problems, checked_claims, audit = (
                                validate_content(
                                    product,
                                    generated,
                                    provider,
                                    brand_profile,
                                    run_ai_audit=True,
                                )
                            )

                            row_result = content_to_row(
                                product,
                                generated,
                                validation_problems,
                                checked_claims,
                                audit,
                                provider,
                                brand_profile,
                            )

                        except Exception as exc:
                            row_result = {
                                **product,
                                "Product Title": "",
                                "Short Description": "",
                                "Long Description": "",
                                "Key Features": "",
                                "Product Benefits": "",
                                "SEO Title": "",
                                "SEO Meta Description": "",
                                "SEO Keywords": "",
                                "Validation Status": "FAILED",
                                "Evidence Checks": 0,
                                "Validation Issues": str(exc),
                                "AI Audit": "FAILED",
                                "AI Engine": provider,
                                "Brand Profile": brand_profile.get(
                                    "brand_name", ""
                                ),
                                "Status": "Failed",
                            }

                        results.append(row_result)
                        progress.progress((index + 1) / len(df))

                    result_df = pd.DataFrame(results)
                    st.session_state["bulk_result_df"] = result_df

                    st.success("✅ Bulk generation completed.")
                    render_bulk_summary(result_df)

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "⬇️ Download Generated Excel",
                        data=dataframe_to_excel(result_df),
                        file_name="generated_product_content.xlsx",
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "spreadsheetml.sheet"
                        ),
                        key="bulk_download_all",
                    )

                    passed_df = result_df[
                        result_df["Validation Status"] == "PASS"
                    ].copy()

                    if not passed_df.empty:
                        st.download_button(
                            "⬇️ Download PASS Products Only",
                            data=dataframe_to_excel(passed_df),
                            file_name="approved_product_content.xlsx",
                            mime=(
                                "application/vnd.openxmlformats-officedocument."
                                "spreadsheetml.sheet"
                            ),
                            key="bulk_download_pass",
                        )

                    render_bulk_review_table(result_df)

        except Exception as exc:
            st.error(f"Could not read the Excel file: {exc}")
