from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv
import os

with open("example.txt","r") as f:
    text=f.read()
    chunks=text.split("\n\n")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings=model.encode(chunks)

chroma_client=chromadb.Client()
collection=chroma_client.create_collection(name="ai_docs")
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=["chunk1","chunk2","chunk3"]
)

question = "How is AI used in healthcare?"
question_embedding = model.encode([question]).tolist()

results = collection.query(
    query_embeddings=question_embedding,
    n_results=2
)

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

context = "\n\n".join(results["documents"][0])

prompt = f"""Answer the question based on the context below.

Context:
{context}

Question:
{question}"""

response = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model="llama-3.3-70b-versatile"
)

print(response.choices[0].message.content)