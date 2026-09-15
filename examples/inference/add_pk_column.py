'''
Run once, rel-amazon does not have a pk identifier for
the review table so it needs to be appended for this
inference run
'''

import duckdb

con = duckdb.connect("rel-amazon.duckdb")

con.execute("""
    ALTER TABLE review ADD COLUMN review_id BIGINT
""")

con.execute("""
    UPDATE review
    SET review_id = rowid
""")

con.execute("""
    CREATE UNIQUE INDEX review_id_idx
    ON review(review_id)
""")

con.close()
