import os
import json
from typing import List
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Template for JSON response
TEMPLATE = """{
    "questions": [
        {
            "id": 1,
            "question": "What is the purpose of assembler directives?",
            "options": [
                "A. To define segments and allocate space for variables",
                "B. To represent specific machine instructions",
                "C. To simplify the programmer's task",
                "D. To provide information to the assembler"
            ],
            "correct_answer": "D. To provide information to the assembler"
        },
        {
            "id": 2,
            "question": "What are opcodes?",
            "options": [
                "A. Instructions for integer addition and subtraction",
                "B. Instructions for memory access",
                "C. Instructions for directing the assembler",
                "D. Mnemonic codes representing specific machine instructions"
            ],
            "correct_answer": "D. Mnemonic codes representing specific machine instructions"
        }
    ]
}"""

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

#     prompt = f"""
#         Act as a teacher and create {num_questions} multiple-choice questions (MCQs) based on the text delimited by four backquotes.
#         Each question should include:
#         - `id`: A unique identifier
#         - `question`: The question text
#         - `options`: A list of four answer options
#         - `correct_answer`: The correct option

#         Example response format:
#         {TEMPLATE}

#         The text is:
#         ````{text}`````
#         Return only the JSON, without any additional text.
#     """
def get_questions(text, num_questions=5):
    prompt = f"""
    Act as a teacher and create {num_questions} MCQs based on the text in four backticks. Return ONLY valid JSON with this structure:
    {TEMPLATE}

    Text: ````{text}`````
    NO EXPLANATIONS, MARKDOWN, OR ADDITIONAL TEXT. JUST THE JSON ARRAY OF QUESTIONS.
    """
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    response = llm.predict(prompt)
    
    print("RAW RESPONSE:", response)  # Debug line
    
    # Clean response (remove markdown syntax)
    clean_response = response.strip().replace("```json", "").replace("```", "")
    
    try:
        # Parse as list directly (since the example has "questions" array)
        questions = json.loads(clean_response)["questions"]
        return questions
    except json.JSONDecodeError as e:
        st.error(f"JSON Error: {e}")
        st.error(f"Cleaned Response Preview: {clean_response[:200]}...")
        return []

def display_questions(questions, expand_answers=False):
    for question in questions:
        st.write(f"### Q{question['id']}: {question['question']}")
        for option in question["options"]:
            st.write(f"- {option}")
        with st.expander("Show answer", expanded=expand_answers):
            st.write(f"**Correct Answer:** {question['correct_answer']}")
        st.divider()
    st.subheader("End of questions")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
    chunks = text_splitter.split_text(text)
    
    chapter_names = []
    for i, chunk in enumerate(chunks):
        if "Chapter" in chunk or "Section" in chunk:
            chapter_name = chunk.split("\n")[0].strip()
        else:
            chapter_name = f"Chapter {i + 1}"
        chapter_names.append(chapter_name)
    
    return chunks, chapter_names

def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def store_questions(questions, chapter):
    with open(f'questions_chapter_{chapter}.json', 'w') as f:
        json.dump(questions, f)

def clean_questions(questions_text: str) -> List[str]:
    lines = questions_text.split("\n")
    cleaned_questions = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            line = line.split(".", 1)[1].strip()
        if line and not line.startswith(("(", "Here are")):
            cleaned_questions.append(line)
    return cleaned_questions[:5]

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_questions(text_chunk: str, chapter: int, chapter_name: str) -> List[str]:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
    
    prompt = PromptTemplate(
        input_variables=["text", "chapter", "chapter_name"],
        template="""
        Based on the following chapter text, generate 5 diverse, thought-provoking questions 
        that assess the user's understanding of the key concepts and encourage real-world application.
        Tailor the questions to follow the natural progression of ideas in the chapter.

        Chapter {chapter}: {chapter_name}
        {text}

        Questions:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    questions_text = chain.run(text=text_chunk, chapter=chapter, chapter_name=chapter_name)
    questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
    return questions[:5]  # Limit to top 5 questions

def generate_personalized_plan(vector_store: FAISS, chapter: int, responses: List[str]):
    plan = []
    for i, response in enumerate(responses):
        results = vector_store.similarity_search_with_score(
            query=response, k=100, filter={"chapter": chapter, "type": "response"}
        )
        question = results[0][0].metadata.get('question', '')
        similarity_score = 1 - results[0][1]  # Convert distance to similarity score
        
        # Dynamic feedback and actions based on response similarity
        if similarity_score < 0.6:
            plan.append(f"Chapter {chapter}: You seem to be struggling with '{question}'. "
                        f"Consider revisiting the chapter.")
        elif 0.6 <= similarity_score < 0.8:
            plan.append(f"Chapter {chapter}: You have a moderate grasp on '{question}'. "
                        f"Try discussing this concept with a peer or writing a reflective summary to solidify your understanding.")
        else:
            plan.append(f"Chapter {chapter}: Excellent understanding of '{question}'. "
                        f"Now you can explore more advanced concepts.")
    return plan

def store_responses(vector_store: FAISS, chapter: int, question: str, response: str):
    response_embedding = embeddings.embed_query(response)
    vector_store.add_texts(
        texts=[response],
        metadatas=[{"chapter": chapter, "question": question, "type": "response"}],
        embeddings=[response_embedding]
    )
    
    # Analyze response and provide feedback based on the similarity score
    similarity_scores = vector_store.similarity_search_with_score(response, k=1)
    top_score = similarity_scores[0][1]  # Best match score
    
    if top_score < 0.6:
        st.write("Your response indicates that you may need more clarity on this topic. Consider revisiting the chapter.")
    elif 0.6 <= top_score < 0.8:
        st.write("You seem to have a fair understanding. To reinforce learning, try applying the concept in a new scenario.")
    else:
        st.write("Great! You have a strong grasp of the topic. Now you can explore more advanced concepts.")

def mcq_generator():
    st.header("MCQ Generator")
    
    with st.sidebar:
        uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])
        num_questions = st.slider("Number of questions", 1, 10, 5)
        expand_answers = st.checkbox("Expand answers by default")

    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        questions = get_questions(text, num_questions)
        display_questions(questions, expand_answers)

def question_generator():
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'text_chunks' not in st.session_state:
        st.session_state.text_chunks = []

    st.subheader("Upload Your PDF Files")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
    
    if st.button("Submit & Process") or st.session_state.processed:
        if not st.session_state.processed:
            with st.spinner("Processing..."):
                # Process the PDF
                raw_text = get_pdf_text(pdf_docs)
                text_chunks, chapter_names = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)  # Create vector store
                
                st.success("PDF processed and embedded successfully!")
                
                # Automatically generate and store questions for each chapter
                for chapter, chunk in enumerate(text_chunks, start=1):
                    questions = generate_questions(chunk, chapter, chapter_names[chapter-1])
                    store_questions(questions, chapter)
                    st.write(f"Generated and stored questions for Chapter {chapter}")

                # Store the processed state and vector store in session state
                st.session_state.processed = True
                st.session_state.vector_store = vector_store
                st.session_state.text_chunks = text_chunks

        # Select chapter to explore
        chapter_selected = st.selectbox("Select a Chapter to Explore", range(1, len(st.session_state.text_chunks) + 1))
        st.subheader(f"Chapter {chapter_selected} Content Preview")
        st.write(st.session_state.text_chunks[chapter_selected - 1][:500] + "...")  # Show first 500 characters of the chapter

        # Load generated questions
        with open(f'questions_chapter_{chapter_selected}.json') as f:
            questions = json.load(f)

        st.subheader("Generated Questions")
        responses = []
        for i, question in enumerate(questions, 1):
            response = st.text_area(f"{i}. {question}")
            responses.append(response)

        if st.button("Submit Responses"):
            for i, response in enumerate(responses):
                store_responses(st.session_state.vector_store, chapter_selected, questions[i], response)

            st.success("Responses submitted!")

            # Generate personalized action plan based on responses
            personalized_plan = generate_personalized_plan(st.session_state.vector_store, chapter_selected, responses)
            st.subheader("Personalized Learning Action Plan")
            for i, action in enumerate(personalized_plan, 1):
                st.write(f"{i}. {action}")

def main():
    st.set_page_config(page_title="PDF Question Generator", page_icon="📚", layout="wide")
    st.title("📚 PDF Question Generator")
    
    app_mode = st.sidebar.selectbox("Select Mode", ["MCQ Generator", "Question Generator"])
    
    if app_mode == "MCQ Generator":
        st.header("MCQ Generator")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        num_questions = st.slider("Number of questions", 1, 10, 5)
        expand_answers = st.checkbox("Show answers")
        
        if uploaded_file:
            text = extract_text_from_pdf(uploaded_file)
            questions = get_questions(text, num_questions)
            display_questions(questions, expand_answers)
    
    elif app_mode == "Question Generator":
        question_generator()

if __name__ == "__main__":
    main()

