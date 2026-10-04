import json
import os
import re
import random
import time
from io import BytesIO

import pandas as pd
import streamlit as st
from ollama import chat as ollama_chat
from google import genai
from google.genai import types

# ---------------- Configuration ----------------
LOCAL_MODEL = "qwen3:4b"
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
]
GEMINI_MAX_RETRIES = 2
GEMINI_INITIAL_BACKOFF_SECONDS = 1.5

st.set_page_config(
    page_title="AI Product Content Studio",
    page_icon="📦",
    layout="wide",
)

# ---------------- Helpers ----------------

def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_gemini_api_key():
    # Streamlit secrets first, then Windows/environment variable.
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


def build_product_prompt(product):
    return f"""
You are an enterprise Product Information Management (PIM) content specialist.

Create e-commerce product content from ONLY the supplied product data.

CRITICAL FACTUALITY AND PROVENANCE RULES:
1. Never invent product facts.
2. Never add specifications, dimensions, weights, percentages, certifications,
   warranties, compatibility, ingredients, performance claims, safety claims,
   origin, standards, or materials unless explicitly supplied.
3. Do not strengthen a claim. For example, "lightweight" must not become
   "ultra-lightweight"; "cotton" must not become "100% cotton".
4. Do not infer facts from the product category or common knowledge.
5. Preserve numbers and units exactly.
6. Every factual claim you make MUST include a source_evidence value copied
   VERBATIM from the supplied product data.
7. If you cannot support a claim with exact source evidence, do not make the claim.
8. Marketing wording is allowed only around supported facts and must not introduce
   a new factual claim.
9. Keep the generated content useful and natural for an e-commerce catalog.

SUPPLIED PRODUCT DATA
Product Name: {product.get("Product Name", "")}
Category: {product.get("Category", "")}
Brand: {product.get("Brand", "")}
Material: {product.get("Material", "")}
Target Customer: {product.get("Target Customer", "")}
Features: {product.get("Features", "")}

Return JSON matching the required schema.

For a claim that is only the product name or category, source_evidence may be the
exact supplied Product Name or Category.

Do not put unsupported claims in the output.
"""


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
    """Return True for errors where retry/fallback is reasonable."""
    code = getattr(exc, "code", None)
    if code in {408, 429, 500, 502, 503, 504}:
        return True

    text = str(exc).lower()
    return any(
        marker in text
        for marker in [
            "503",
            "unavailable",
            "service unavailable",
            "high demand",
            "temporarily overloaded",
            "429",
            "resource exhausted",
            "500",
            "502",
            "504",
            "deadline exceeded",
            "timeout",
        ]
    )


def gemini_error_message(exc):
    code = getattr(exc, "code", None)
    if code:
        return f"HTTP {code}: {exc}"
    return str(exc)


def generate_with_gemini(client, prompt, schema):
    """Generate JSON with retry + model fallback for transient Gemini failures."""
    failures = []

    for model_index, model in enumerate(GEMINI_MODELS):
        for attempt in range(GEMINI_MAX_RETRIES + 1):
            try:
                # Gemini 3.8 Flash does not use temperature/top_p/top_k.
                # Keep the config limited to structured JSON output.
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

                message = gemini_error_message(exc)
                failures.append(f"{model} attempt {attempt + 1}: {message}")

                # Retry the same model with exponential backoff + jitter.
                if attempt < GEMINI_MAX_RETRIES:
                    delay = GEMINI_INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    delay += random.uniform(0, 0.5)
                    time.sleep(delay)

        # If this model is exhausted, move to the next stable Flash model.
        if model_index < len(GEMINI_MODELS) - 1:
            st.warning(
                f"Gemini model `{model}` is temporarily unavailable. "
                f"Trying fallback model `{GEMINI_MODELS[model_index + 1]}`..."
            )

    details = " | ".join(failures[-6:])
    raise RuntimeError(
        "Gemini could not complete the request after retries and fallback models. "
        f"Details: {details}"
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
        client = get_gemini_client()
        return generate_with_gemini(client, prompt, schema)

    raise ValueError(f"Unsupported AI engine: {provider}")


def generate_content(product, provider):
    return generate_json(
        provider,
        build_product_prompt(product),
        content_schema(),
        temperature=0.1,
    )


def source_text(product):
    return " ".join(
        clean_value(v)
        for v in product.values()
        if clean_value(v)
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
            return

        if evidence not in source:
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


def build_audit_prompt(product, output):
    return f"""
You are a strict PIM content quality auditor.

Compare SOURCE DATA against GENERATED CONTENT.

SOURCE DATA:
{json.dumps(product, ensure_ascii=False, indent=2)}

GENERATED CONTENT:
{json.dumps(output, ensure_ascii=False, indent=2)}

Identify only factual claims in the generated content that are unsupported,
strengthened beyond the source, or contradicted by the source.

Examples of failures:
- Source says "cotton"; generated says "100% cotton".
- Source says "lightweight"; generated says "ultra-lightweight".
- Source contains no waterproof claim; generated says "waterproof".
- Source says EVA sole; generated says shock-absorbing EVA sole unless shock
  absorption was explicitly supplied.

Return PASS only if there are no meaningful factual overclaims.
"""


def audit_content_with_model(product, output, provider):
    return generate_json(
        provider,
        build_audit_prompt(product, output),
        audit_schema(),
        temperature=0.0,
    )


def validate_content(product, output, provider, run_ai_audit=True):
    problems, checked = validate_provenance(product, output)
    problems.extend(validate_required_evidence(output))

    unexpected_numbers = validate_numbers(product, output)
    if unexpected_numbers:
        problems.append(
            "Potential unsupported numeric values detected: "
            + ", ".join(unexpected_numbers)
        )

    audit = None
    if run_ai_audit:
        try:
            audit = audit_content_with_model(product, output, provider)
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


def flatten_provenance(items):
    values = []
    for item in normalize_list(items):
        if isinstance(item, dict):
            text = clean_value(item.get("text"))
            evidence = clean_value(item.get("source_evidence"))
            if text:
                values.append(text)
            if evidence:
                values.append(f"[Evidence: {evidence}]")
        else:
            values.append(str(item))
    return " | ".join(values)


def normalize_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [str(value)]


def content_to_row(product, result):
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
        "Content Warnings": " | ".join(normalize_list(result.get("content_warnings"))),
    }


def dataframe_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Generated Content")
    output.seek(0)
    return output


def render_validation(validation_problems, checked_claims, audit):
    if validation_problems:
        st.error(
            f"⚠️ Validation found {len(validation_problems)} issue(s). "
            "Review before publishing."
        )
        for warning in validation_problems:
            st.write(f"- {warning}")
    else:
        st.success(
            f"✅ Validation passed. {checked_claims} factual item(s) have source evidence."
        )

    if audit:
        if audit.get("status") == "PASS":
            st.success("🤖 Independent AI factuality audit: PASS")
        else:
            st.warning("🤖 Independent AI factuality audit: FAIL/REVIEW")


# ---------------- UI ----------------

st.title("📦 AI Product Content Studio")
st.caption("Product content generation with local Qwen3 4B or cloud Gemini")

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
        st.caption("Primary: gemini-3.8-flash | Fallbacks: gemini-3.7-flash → gemini-3.6-flash")
    else:
        st.success("Local mode selected")
        st.caption("Model: qwen3:4b")

    st.divider()
    st.caption(
        "The same provenance and factuality validation runs after generation "
        "regardless of the selected model."
    )

tab_single, tab_bulk = st.tabs(["Single Product", "Bulk Excel"])

# ---------------- Single Product ----------------

with tab_single:
    st.subheader("Generate content for one product")

    left, right = st.columns(2)

    with left:
        product_name = st.text_input("Product Name", key="single_name")
        category = st.text_input("Category", key="single_category")
        brand = st.text_input("Brand", key="single_brand")
        target_customer = st.text_input("Target Customer", key="single_customer")

    with right:
        material = st.text_input("Material", key="single_material")
        features = st.text_area(
            "Product Features",
            height=140,
            placeholder="Example: Lightweight, breathable mesh, EVA cushioning, non-slip outsole",
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

            engine_name = "Gemini" if provider.startswith("Cloud") else "Qwen3 4B"
            with st.spinner(f"Generating with {engine_name}..."):
                try:
                    result = generate_content(product, provider)

                    with st.spinner("Running provenance and factuality checks..."):
                        validation_problems, checked_claims, audit = validate_content(
                            product, result, provider, run_ai_audit=True
                        )

                    st.divider()
                    st.subheader("Generated Content")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### Product Title")
                        st.write(result.get("product_title", ""))
                        st.markdown("### Short Description")
                        st.write(result.get("short_description", ""))
                        st.markdown("### Long Description")
                        st.write(result.get("long_description", ""))
                        st.markdown("### Key Features")
                        for item in normalize_list(result.get("key_features")):
                            if isinstance(item, dict):
                                st.markdown(f"- {item.get('text', '')}")
                                st.caption(f"Evidence: {item.get('source_evidence', '')}")
                            else:
                                st.markdown(f"- {item}")

                    with c2:
                        st.markdown("### Product Benefits")
                        for item in normalize_list(result.get("product_benefits")):
                            if isinstance(item, dict):
                                st.markdown(f"- {item.get('text', '')}")
                                st.caption(f"Evidence: {item.get('source_evidence', '')}")
                            else:
                                st.markdown(f"- {item}")
                        st.markdown("### SEO Title")
                        st.write(result.get("seo_title", ""))
                        st.markdown("### SEO Meta Description")
                        st.write(result.get("seo_meta_description", ""))
                        st.markdown("### SEO Keywords")
                        st.write(", ".join(normalize_list(result.get("seo_keywords"))))

                    render_validation(validation_problems, checked_claims, audit)
                    if provider.startswith("Cloud"):
                        used_model = st.session_state.get("last_gemini_model", "unknown")
                        if st.session_state.get("last_gemini_fallback"):
                            st.info(f"Gemini fallback used: {used_model}")
                        else:
                            st.caption(f"Gemini model used: {used_model}")
                    st.info(
                        "The tool uses deterministic evidence/numeric checks plus a separate AI audit. "
                        "It is a review aid, not a legal or publication guarantee."
                    )

                except Exception as e:
                    st.error(f"Generation failed: {e}")

# ---------------- Bulk Excel ----------------

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
                "Features": "Lightweight, breathable mesh, EVA cushioning, non-slip outsole",
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

    uploaded_file = st.file_uploader("Upload product Excel file", type=["xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            required = [
                "Product Name", "Category", "Brand", "Material", "Target Customer", "Features"
            ]
            missing = [column for column in required if column not in df.columns]

            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                st.success(f"Loaded {len(df)} products.")
                st.dataframe(df.head(10), use_container_width=True)

                generate_bulk = st.button(
                    "🚀 Generate Content for All Products",
                    type="primary",
                )

                if generate_bulk:
                    results = []
                    progress = st.progress(0)
                    status = st.empty()

                    for index, row in df.iterrows():
                        product = {column: clean_value(row[column]) for column in required}
                        status.write(
                            f"Processing product {index + 1} of {len(df)}: {product['Product Name']}"
                        )

                        try:
                            generated = generate_content(product, provider)
                            validation_problems, checked_claims, audit = validate_content(
                                product, generated, provider, run_ai_audit=True
                            )

                            row_result = content_to_row(product, generated)
                            row_result["Validation Status"] = (
                                "PASS" if not validation_problems else "REVIEW"
                            )
                            row_result["Evidence Checks"] = checked_claims
                            row_result["Validation Issues"] = " | ".join(validation_problems)
                            row_result["AI Audit"] = (
                                audit.get("status", "UNKNOWN")
                                if isinstance(audit, dict) else "UNKNOWN"
                            )
                            row_result["AI Engine"] = provider
                            row_result["Status"] = (
                                "Generated" if not validation_problems else "Generated - Review"
                            )

                        except Exception as e:
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
                                "Content Warnings": str(e),
                                "Validation Status": "FAILED",
                                "Evidence Checks": 0,
                                "Validation Issues": str(e),
                                "AI Audit": "FAILED",
                                "AI Engine": provider,
                                "AI Model": (st.session_state.get("last_gemini_model", "") if provider.startswith("Cloud") else LOCAL_MODEL),
                                "Status": "Failed",
                            }

                        results.append(row_result)
                        progress.progress((index + 1) / len(df))

                    result_df = pd.DataFrame(results)
                    st.success("✅ Bulk generation completed.")
                    st.dataframe(result_df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download Generated Excel",
                        data=dataframe_to_excel(result_df),
                        file_name="generated_product_content.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

        except Exception as e:
            st.error(f"Could not read the Excel file: {e}")
