from pathlib import Path
import databases
from fastapi import FastAPI
from main import get_embedding

api = FastAPI()
db = databases.Databases(directory=Path("/home/diego/Downloads/DeepQuestions"))

@api.get("/search")
def search_server(query: str):
    """
    Search for a query in a directory of transcriptions.
    """

    emb = get_embedding([query])
    return db.search(emb[0])
