from typing import AnyStr, Optional, Type, Any
from gentopia.tools.basetool import BaseTool
from pydantic import BaseModel, Field
import requests
import fitz  # PyMuPDF
import io

class DocArgs(BaseModel):
    qry: str = Field(..., description="Please enter the URL of the PDF document you want to read.")

class Document(BaseTool):
    name = "document_reader"
    description = "Please provide the URL of the PDF document for the reader to access."
    args_schema: Optional[Type[BaseModel]] = DocArgs

    def _run(self, qry: AnyStr) -> str:
        try:
            # Use requests to fetch the PDF
            response = requests.get(qry, headers={'User-Agent': "Magic Browser"})
            response.raise_for_status()  # Raise error for bad status codes

            # Read the file in bytes
            pdf_bytes = io.BytesIO(response.content)
            
            # Open the PDF using PyMuPDF (fitz)
            document_reader = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Extract text from all pages
            extracted_text = ""
            for page_num in range(len(document_reader)):
                page = document_reader.load_page(page_num)
                extracted_text += page.get_text("text") + "\n\n"

            return extracted_text
        
        except Exception as error:
            raise ValueError("Unable to access the PDF. Please check if the URL is correct and leads to a valid PDF.") from error

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

if __name__ == "__main__":
    document_reader = Document()
    try:
        ans = document_reader._run("https://arxiv.org/pdf/2201.05966.pdf")
        print(ans)
    except ValueError as error:
        print(error)
