import json
import requests
from typing import Dict
from loguru import logger


def clean_ocr_text(raw_text: str) -> str:
    """
    Clean OCR text to remove unwanted patterns.
    """
    lines = raw_text.splitlines()
    cleaned_lines = [line for line in lines if not line.startswith("CAN")]
    return "\n".join(cleaned_lines)


def send_prompt_to_llm(prompt, model="llama2", endpoint="https://ttuiiwtjvxy3jxw2.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions"):
    try:
        response = requests.post(
            endpoint,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error contacting LLM: {e}")
        return '{"error": "Failed to reach LLM service"}'


def generate_structured_data_with_llama2(raw_text: str, document_type: str) -> Dict:
    """
    Process raw text using an LLM endpoint for better structured data extraction.
    """
    try:
        logger.info(f"Generating structured data using LLM for {document_type}...")

        raw_text = clean_ocr_text(raw_text)

        # Introduction prompt for setting context
        introduction_prompt = """
        You are an AI model designed to process and label the extracted data.
        Your task is to extract specific fields from the given OCR text and return **only** the valid **JSON structure** without any additional introduction ,text, explanations, or comments.
        Your output must generate **only JSON** with the exact field names required. Any missing or unknown field should be left as an empty string `""`.
        Ensure dates are formatted as `dd.mm.yyyy` and are realistic.
        dont put this in the output 
        """

        # Create the document-specific prompt with context
        if document_type == "id_card":
            prompt = f"""
            {introduction_prompt}
            Process the following ID card text and return only the **JSON structure**:
            {raw_text}

            **Output JSON format (do not add anything else, follow the structure exactly)**:
            {{
                "Prénom": "string",
                "Nom de famille": "string",
                "Date de naissance": "date",
                "Lieu de naissance": "string",
                "Num d'identité": "string",
                "Valable jusqu'au": "date"
            }}
            """

        elif document_type == "passport":
            prompt = f"""
            {introduction_prompt}
            Process the following passport text and return only the **JSON structure**:
            {raw_text}

            **Output JSON format (do not add anything else, follow the structure exactly)**:
            {{
                "Nom": "string",
                "Prénom": "string",
                "Date de naissance": "date",
                "Nationalité": "string",
                "Numéro de passeport": "string",
                "Date d'expiration": "date"
            }}
            """

        elif document_type == "drivers_license":
            prompt = f"""
            {introduction_prompt}
            Process the following driver's license text and return only the **JSON structure**:
            {raw_text}

            **Output JSON format (do not add anything else, follow the structure exactly)**:
            {{
                "Nom": "string",
                "Prénom": "string",
                "Date de naissance": "date",
                "Numéro de permis": "string",
                "Date d'expiration": "date",
                "Adresse": "string"
            }}
            """

        else:
            raise ValueError("Unknown document type")

        structured_data_content = send_prompt_to_llm(prompt)

        # Log the raw response for debugging purposes
        logger.debug(f"Raw response from LLM: {structured_data_content}")

        if structured_data_content == '{"error": "Failed to reach LLM service"}':
            logger.error(f"Failed to reach LLM service for {document_type}.")
            return {"error": "Failed to reach LLM service"}

        try:
            # Strip any unwanted introductory or explanatory text before JSON parsing
            structured_data_content = structured_data_content.strip()
            if structured_data_content.startswith("Here is the output JSON structure") or not structured_data_content.startswith("{"):
                logger.error(f"Unexpected response format for {document_type}. Raw response: {structured_data_content}")
                return {"error": "Unexpected response format"}

            # Attempt to parse the response as JSON
            structured_data = json.loads(structured_data_content)

            # Ensure the generated JSON has the necessary fields
            if document_type == "id_card":
                required_fields = ["Prénom", "Nom de famille", "Date de naissance", "Lieu de naissance", "Num d'identité", "Valable jusqu'au"]
            elif document_type == "passport":
                required_fields = ["Nom", "Prénom", "Date de naissance", "Nationalité", "Numéro de passeport", "Date d'expiration"]
            elif document_type == "drivers_license":
                required_fields = ["Nom", "Prénom", "Date de naissance", "Numéro de permis", "Date d'expiration", "Adresse"]
            else:
                required_fields = []

            # Check that all required fields are in the JSON
            if all(field in structured_data for field in required_fields):
                logger.info(f"LLM generated valid data for {document_type}: {structured_data}")
                return structured_data
            else:
                logger.error(f"Missing required fields in JSON for {document_type}. Raw response: {structured_data_content}")
                return {"error": "Generated JSON is missing required fields"}

        except json.JSONDecodeError:
            logger.error(f"JSON decoding failed for {document_type}. Raw response: {structured_data_content}")
            return {"error": "LLM did not return valid JSON"}

    except Exception as e:
        logger.error(f"Error processing text for {document_type}: {str(e)}")
        return {"error": f"Failed to process text for {document_type}"}
