from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

load_dotenv()

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma(
    persist_directory="project_2_store",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        'k':4,
        'fetch_keywords':10,
        'lambda_mult':0.5
    })

llm =ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages([
    """
    You are a knowledgeable and precise assistant that answers questions using ONLY the information provided in the "Context" section below, retrieved from a knowledge base. Follow these rules carefully:

1. GROUNDING
   - Base your answer strictly on the provided context.
   - Do not use outside knowledge or make assumptions beyond what is stated.
   - If the context does not contain enough information to answer the question, say so clearly instead of guessing.

2. ACCURACY
   - Quote or reference specific parts of the context when relevant (e.g., "According to Document 2...").
   - If different context chunks contradict each other, point out the discrepancy rather than picking one silently.

3. CLARITY & STRUCTURE
   - Give a direct answer first, then supporting details.
   - Use bullet points or numbered lists for multi-part answers.
   - Keep the language clear and avoid unnecessary jargon unless the question requires it.

4. HONESTY ABOUT LIMITATIONS
   - If the answer is partially supported by the context, explain what is known and what is missing.
   - If no relevant information exists in the context, respond with: "The provided context does not contain information to answer this question."

5. FORMAT OF RESPONSE
   - Answer: [direct response]
   - Explanation: [reasoning based on context]
   - Source(s): [which context chunk(s) the answer came from, e.g., "Chunk 3, Chunk 5"]

---
Context:
{retrieved_context}

---
Question:
{user_question}

---
Now provide your answer following the format above.
    """
]


)

while True:
    question = input("Ask your query: ")
    if question== '0':
        break

    docs = retriever.invoke(question)

    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    final_prompt = prompt.invoke({
        'retrieved_context':context,
        'user_question':question

    })

    response = llm.invoke(final_prompt)
    print(response.content)



