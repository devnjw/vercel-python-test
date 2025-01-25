import requests
import json
import httpx
from typing import List, Literal, Annotated, Optional
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI()

DOCUMENT_PARSE_BASE_URL = "https://api.upstage.ai/v1/document-ai/document-parse"
DEFAULT_NUM_PAGES = 10
DOCUMENT_PARSE_DEFAULT_MODEL = "document-parse"
OCR = Literal["auto", "force"]
SplitType = Literal["none", "page", "element"]
OutputFormat = Literal["text", "html", "markdown"]


@app.post("/api")
def multi_model(
    messages: Annotated[str, Form()],
    model: Annotated[str, Form(...)],
    documents: List[UploadFile] = File(...),
    authorization: Annotated[str, Header()] = None,
    stream: Annotated[bool, Form()] = False,
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    api_key = authorization.split("Bearer ")[1]
    messages = json.loads(messages)
    doc_contents = parse_documents(documents, api_key)
    messages = [{"role": "system", "content": doc_contents}] + messages

    if stream:
        return StreamingResponse(
            stream_chat_completion(messages, model, api_key),
            media_type="application/json",
        )
    else:
        return chat_completion(messages, model, api_key)


def parse_documents(documents: List[UploadFile], api_key: str) -> str:
    doc_contents = "Documents:\n"
    for doc in documents:
        response = requests.post(
            DOCUMENT_PARSE_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={
                "document": doc.file,
            },
        )
        elements = response.json().get("elements", [])
        result = ""
        for element in elements:
            result += parse_dp_output(element, "html")

        doc_contents += f"{doc.filename}:\n{result}\n\n"

    return doc_contents


def parse_dp_output(data: dict, output_format: OutputFormat) -> str:
    content = data["content"]
    if output_format == "text":
        return content["text"]
    elif output_format == "html":
        return content["html"]
    elif output_format == "markdown":
        return content["markdown"]
    else:
        raise ValueError(f"Invalid output type: {output_format}")


async def stream_chat_completion(messages: List[dict], model: str, api_key: str):
    url = 'https://api.upstage.ai/v1/solar/chat/completions'
    headers = {
        'authorization': f'Bearer {api_key}',
        'content-type': 'application/json',
    }
    data = {
        "messages": messages,
        "model": model,
        "stream": True,
    }

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, headers=headers, json=data) as response:
            async for chunk in response.aiter_text():
                yield chunk


def chat_completion(messages: List[dict], model: str, api_key: str):
    url = 'https://api.upstage.ai/v1/solar/chat/completions'
    headers = {
        'authorization': f'Bearer {api_key}',
        'content-type': 'application/json',
    }
    data = {
        "messages": messages,
        "model": model,
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()
