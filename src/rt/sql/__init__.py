"""RT-J contexts straight from a database: SQL picks the rows, no preprocessing.

    con = duckdb.connect("mini-shop.duckdb", read_only=True)
    db = rt.sql.load(con)
    enc = rt.sql.Encoder(con, db)
    ctx = rt.sql.Retriever(con, db, enc).build("customer-churn", "test", 0)
    batch = rt.sql.to_batch([ctx], rt.sql.TextEmbedder("all-MiniLM-L12-v2"), 1024, 384)
    ctx.trace()  # every row, why it is there, and every query with its timing

`meta` reads the database, `encode` turns rows into cells as rustler's
preprocessor does, `retrieve` chooses rows with SQL, and `context` holds the
result and builds the model's tensors.
"""

from rt.sql.context import Context, Node, to_batch
from rt.sql.encode import Encoder, TextEmbedder
from rt.sql.meta import Database, load
from rt.sql.retrieve import RetrieveConfig, Retriever

__all__ = [
    "Context",
    "Database",
    "Encoder",
    "Node",
    "RetrieveConfig",
    "Retriever",
    "TextEmbedder",
    "load",
    "to_batch",
]
