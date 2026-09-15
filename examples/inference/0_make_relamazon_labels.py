import duckdb
con = duckdb.connect("rel-amazon.duckdb", read_only=True)

# create review_id PK over review table rows
con.execute("""
    CREATE OR REPLACE TABLE review AS
    SELECT
    row_number() OVER (
        ORDER BY review_time, customer_id, product_id
    ) - 1 AS review_id,
    *
    FROM review;
    """)
reviews = con.execute("""
    SELECT review_id, review_time, rating
    FROM review
    WHERE rating IS NOT NULL
    ORDER BY review_time
    """).df()

n = len(reviews)
train = reviews.iloc[: int(n*0.8)]
val = reviews.iloc[int(n * 0.8) : int(n * 0.9)]
test = reviews.iloc[int(n*0.9) :]

out = "examples/inference/out/labels"
train.to_parquet(f"{out}/review_rating_train.parquet", index=False)
val.to_parquet(f"{out}/review_rating_val.parquet", index=False)
test.to_parquet(f"{out}/review_rating_test.parquet", index=False)

