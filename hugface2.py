from typing import Literal, Optional, Any, Dict
from pydantic import BaseModel, Field
import json
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import re
import json
import os
from dotenv import load_dotenv
# Load the .env file
load_dotenv()

# Access the variable
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# -----------------------
# Helpers (edge cases)
# -----------------------
def _normalize_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().strip('"').strip("'")
        if s.lower() in {"none", "null", ""}:
            return None
        return s
    s = str(v).strip()
    if s.lower() in {"none", "null", ""}:
        return None
    return s




ALLOWED_CATEGORIES = [
    "Sexism",
    "Xenophobia",
    "Ableism",
    "Racism",
    "Religious hate",
    "Cultural discrimination",
    "Bullying",
]


def _sanitize_category(raw: Any) -> Optional[str]:
    s = _normalize_str(raw)
    if s is None:
        return None

    s_clean = " ".join(s.split()).strip()
    s_low = s_clean.lower()

    synonym_map = {
        "racism": "Racism",
        "racial hate": "Racism",
        "anti-black": "Racism",
        "sexism": "Sexism",
        "misogyny": "Sexism",
        "misandry": "Sexism",
        "xenophobia": "Xenophobia",
        "anti-immigrant": "Xenophobia",
        "ableism": "Ableism",
        "disability hate": "Ableism",
        "religious hate": "Religious hate",
        "islamophobia": "Religious hate",
        "antisemitism": "Religious hate",
        "anti-semitism": "Religious hate",
        "christianophobia": "Religious hate",
        "cultural discrimination": "Cultural discrimination",
        "cultural hate": "Cultural discrimination",
        "bullying": "Bullying",
        "harassment": "Bullying",
        "insult": "Bullying",
        "abuse": "Bullying",
    }

    mapped = synonym_map.get(s_low)
    if mapped in ALLOWED_CATEGORIES:
        return mapped

    for c in ALLOWED_CATEGORIES:
        if s_low == c.lower():
            return c

    return None


def _coerce_sentiment(v: Any) -> str:
    s = (_normalize_str(v) or "").lower()
    mapping = {
        "pos": "positive",
        "positive": "positive",
        "neutral": "neutral",
        "neg": "negative",
        "negative": "negative",
        "mixed": "neutral",
        "unknown": "neutral",
        "other": "neutral",
        "n/a": "neutral",
        "na": "neutral",
    }
    return mapping.get(s, "neutral")




def _safe_invoke(chain, payload: Dict[str, Any], stage: str, max_retries: int = 3) -> Dict[str, Any]:
    txt = payload.get("text", "")
    
    for attempt in range(max_retries):
        try:
            # 1. Attempt LLM invocation
            response = chain.invoke(payload)
            
            # Convert response to string for regex cleaning if it's not already a dict
            raw_str = str(response) if not isinstance(response, dict) else ""

            # 2. LEVEL 1 SECURITY: Regex Extraction
            # This finds everything between the first '{' and last '}'
            if not isinstance(response, dict):
                match = re.search(r'(\{.*\}|\[.*\])', raw_str, re.DOTALL)
                if match:
                    try:
                        response = json.loads(match.group(1))
                    except json.JSONDecodeError:
                        raise ValueError(f"Regex found segment but JSON is invalid: {match.group(1)}")
                else:
                    raise ValueError(f"No JSON structure found in response: {raw_str}")

            # 3. LEVEL 2 SECURITY: Flattening Nested Dicts
            if isinstance(response, dict):
                clean_response = {}
                for k, v in response.items():
                    # Fixes {"sentiment": {"value": "negative"}} -> {"sentiment": "negative"}
                    if isinstance(v, dict) and "value" in v:
                        clean_response[k] = v["value"]
                    else:
                        clean_response[k] = v
                return clean_response

        except Exception as e:
            print(f"[{stage}] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                continue 

    # 4. LEVEL 3 SECURITY: Final Fallbacks
    print(f"[{stage}] All retries failed. Returning safe defaults.")
    if stage == "CLASSIFY":
        return {"sentiment": "neutral", "category": None}
    if stage == "TRANSFORM":
        return {"originalStatement": txt, "replacement": txt}
    
    return {}


# -----------------------
# Your requested edge-case validators (no logging)
# -----------------------

def _validate_and_normalize_classification(raw: Dict[str, Any]) -> Dict[str, Any]:
    sentiment = _coerce_sentiment(raw.get("sentiment"))
    raw_cat = raw.get("category")
    
    # If LLM returned the string "None", treat it as Python None
    if isinstance(raw_cat, str) and raw_cat.lower() == "none":
        category = None
    else:
        category = _sanitize_category(raw_cat)
    
    # Your existing edge-case logic remains a great safety net
    if sentiment == "negative" and not category:
        category = "Bullying"

    return {"sentiment": sentiment, "category": category}

def _validate_transform(raw: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    original = _normalize_str(raw.get("originalStatement")) or user_text
    replacement = _normalize_str(raw.get("replacement"))
    
    # Extract and normalize severity
    severity = _normalize_str(raw.get("severity"))
    if severity not in ["low", "medium", "high"]:
        severity = "medium"  # Default fallback

    if not replacement or "cannot" in replacement.lower() or "rephrase" in replacement.lower():
        replacement = "I disagree with this perspective." 

    return {
        "originalStatement": original, 
        "replacement": replacement,
        "severity": severity
    }
# -----------------------
# Schemas
# -----------------------
class Classification(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="The emotional tone of the text."
    )
    category: Literal[
        "Sexism", "Xenophobia", "Ableism", "Racism", 
        "Religious hate", "Cultural discrimination", "Bullying", "None"
    ] = Field(
        description="The category of hate or offense. Return 'None' if the text is positive or neutral."
    )

class DataSchema(BaseModel):
    originalStatement: str = Field(
        description="The exact original text provided by the user."
    )
    severity: Optional[Literal["low", "medium", "high"]] = Field(
        description="The severity level of the original statement. Optional field." 
    )
    replacement: str = Field(
        description="The rewritten version that is professional and free of insults. "
                    "This must be the actual rewritten text, not a suggestion about the text."
    )

# -----------------------
# LLM setup
# -----------------------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.0,
    huggingfacehub_api_token=hf_token,
)
chat_model = ChatHuggingFace(llm=llm)

# -----------------------
# Step 1: Classification chain
# -----------------------
class_parser = JsonOutputParser(pydantic_object=Classification)
class_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a content moderation expert. Analyze the text and provide a classification. "
     "You MUST include both 'sentiment' and 'category' in your JSON response. "
     "If the text does not fit a hate category, use 'None'. "
     "\n{format_instructions}"),
    ("human", "{text}")
]).partial(format_instructions=class_parser.get_format_instructions())

class_chain = class_prompt | chat_model | class_parser

# -----------------------------
# Step 2: Transformation chain
# ----------------------------
trans_parser = JsonOutputParser(pydantic_object=DataSchema)

trans_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional editor and content analyst. "
     "1. Analyze the 'severity' of the bullying: 'low' (mild insults), 'medium' (harsh/personal), or 'high' (threats/severe harassment)."
     "\n2. Rewrite toxic 'Bullying' statements into neutral, constructive language."
     "\n\nRules:"
     "\n- Preserve the core meaning but remove all insults."
     "\n- If the text is a pure insult, rewrite as a statement of disagreement."
     "\n- Return ONLY valid JSON."
     "\n\n{format_instructions}"),
    ("human", "{text}")
]).partial(format_instructions=trans_parser.get_format_instructions())


trans_chain = trans_prompt | chat_model | trans_parser



# -----------------------
# Workflow Function (minimal debug prints)
# -----------------------
def process_message(user_text: str) -> Dict[str, str]:
    print("\n==============================")
    print("INPUT:")
    print(user_text)

    # 1) CLASSIFY: show raw first LLM output
    raw_class = _safe_invoke(class_chain, {"text": user_text}, stage="CLASSIFY")
    print("\nCLASSIFY (first LLM output):")
    print("llm response" , raw_class)

    # 2) Normalize with your validators
    classification = _validate_and_normalize_classification(raw_class)
    sentiment = classification["sentiment"]
    category = classification["category"]

    # Branch 1: non-negative OR no category
    if sentiment in {"positive", "neutral"} or category is None:
        print("\nBRANCH 1")
        result = {
            "sentiment": sentiment,
            "category": category,
            "originalStatement": user_text,
            "replacement": user_text,
        }
        print("\nOUTPUT (original + replacement):")
        print(result)
        return result

    # Branch 2: Hate categories -> blank replacement
    hate_categories = {
        "Racism", "Sexism", "Xenophobia", "Ableism", "Religious hate", "Cultural discrimination"
    }
    if sentiment == "negative" and category in hate_categories:
        print("\nBRANCH 2")
        result = {
            "sentiment": sentiment,
            "category": category,
            "originalStatement": user_text,
            "replacement": "",
        }
        print("\nOUTPUT (original + replacement):")
        print(result)
        return result

    # Branch 3: Bullying -> transform
    if sentiment == "negative" and category == "Bullying":
        print("\nBRANCH 3")

        raw_trans = _safe_invoke(trans_chain, {"text": user_text}, stage="TRANSFORM")
        trans = _validate_transform(raw_trans, user_text)

        # 'trans' now contains 'severity', 'originalStatement', and 'replacement'
        result = {"sentiment": sentiment, "category": category} | trans
        print("\nOUTPUT (original + replacement + severity):")
        print(result)
        return result
    

    # Fallback
    print("\nBRANCH FALLBACK")
    result = {
        "sentiment": sentiment,
        "category": category,
        "originalStatement": user_text,
        "replacement": user_text,
    }
    print("\nOUTPUT (original + replacement):")
    print(result)
    return result


# -----------------------
# Test
# -----------------------
if __name__ == "__main__":
    text_input = "i think you are an idiot and your ideas are trash and i wish death upon you"
    final_result = process_message(text_input)
    print("\n--- FINAL WORKFLOW OUTPUT ---")
    print(json.dumps(final_result, ensure_ascii=False, indent=2))
