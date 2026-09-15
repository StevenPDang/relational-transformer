import duckdb
from pathlib import Path


_HERE = Path(__file__).resolve().parent
con = duckdb.connect(str(_HERE.parents[1] / "rel-amazon.duckdb"), read_only=True)

reviews = con.execute("""
    SELECT review_id, review_time, rating
    FROM review
    WHERE rating IS NOT NULL
    ORDER BY review_time
    """).df()

n = len(reviews)
train = reviews.iloc[: int(n * 0.8)]
val = reviews.iloc[int(n * 0.8) : int(n * 0.9)]
test = reviews.iloc[int(n * 0.9) :]

out = _HERE / "out" / "labels"
out.mkdir(parents=True, exist_ok=True)
train.to_parquet(out / "review_rating_train.parquet", index=False)
val.to_parquet(out / "review_rating_val.parquet", index=False)
test.to_parquet(out / "review_rating_test.parquet", index=False)

con.close()
