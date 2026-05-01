from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

def llm_judge(resume_text, predicted_role):

    template = """
    You are an expert career evaluator.

    Resume:
    {resume_text}

    Predicted Role: {predicted_role}

    Evaluate suitability.

    Answer strictly:
    Verdict: YES or NO
    Reason: <short reason>
    """

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({
            "resume_text": resume_text[:3000],  # prevent token overflow
            "predicted_role": predicted_role
        })

        lines = response.upper().split("\n")
        verdict_line = next((l for l in lines if "VERDICT" in l), "")

        if "YES" in verdict_line:
            verdict = 1
        elif "NO" in verdict_line:
            verdict = 0
        else:
            verdict = -1

        return verdict, response

    except Exception as e:
        return -1, str(e)