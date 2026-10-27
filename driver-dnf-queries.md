# rt.sql queries on rel-f1 / driver-dnf

Query shapes from DuckDB's query log (`predict --query-log`), counted over the
run. In the SQL, `N` stands for a number written into the query (a time bound,
a timestamp divisor, `bfs_width`, an entity id) and `?` is a bound parameter
(the list of row ids or FK values for the BFS level).

## Summary

| Count | Step | Table | Through |
| ---: | --- | --- | --- |
| 38,075 | parents | `drivers` | FK from results, qualifying, standings, task rows |
| 7,119 | children | `qualifying` | `driverId` |
| 7,119 | children | `results` | `driverId` |
| 7,119 | children | `standings` | `driverId` |
| 7,119 | children | `driver-dnf/train` | `driverId` |
| 7,119 | children | `driver-dnf/val` | `driverId` |
| 7,119 | children | `driver-dnf/test` | `driverId` |
| 1,930 | parents | `races` | `raceId` |
| 1,886 | parents | `constructors` | `constructorId` |
| 1,093 | children | `constructor_results` | `constructorId` |
| 1,093 | children | `constructor_standings` | `constructorId` |
| 1,093 | children | `qualifying` | `constructorId` |
| 1,093 | children | `results` | `constructorId` |
| 719 | example ranking (one page) | all `driver-dnf` splits | — |
| 542 | example fetch | `driver-dnf/val` | — |

## Queries

### 1. Parents: `drivers` (38,075)

```sql
SELECT
    __row,
    "driverRef", "code", "forename", "surname",
    epoch_ns("dob"),
    "nationality",
    NULL::BIGINT                                  -- no time column
FROM "drivers"
WHERE __row IN (SELECT unnest(?::BIGINT[]))       -- parent ids from the BFS level
  AND TRUE                                        -- no time filter: untimestamped table
```

### 2. Children of drivers: `qualifying` (7,119)

```sql
SELECT
    __row,
    "raceId", "driverId", "constructorId",
    "number", "position",
    epoch_ns("date"),
    epoch_ns("date") // N,                        -- row timestamp, seconds
    "driverId" AS __parent
FROM "qualifying"
WHERE "driverId" IN (SELECT unnest(?::BIGINT[]))
  AND (epoch_ns("date") // N IS NULL
       OR epoch_ns("date") // N <= N)             -- not after the seed's time
QUALIFY row_number() OVER (
    PARTITION BY "driverId"
    ORDER BY epoch_ns("date") // N DESC NULLS LAST, __row DESC
) <= N                                            -- bfs_width most recent per driver
```

### 3. Children of drivers: `results` (7,119)

```sql
SELECT
    __row,
    "raceId", "driverId", "constructorId",
    "number", "grid", "position", "positionOrder", "points",
    "laps", "milliseconds", "fastestLap", "rank", "statusId",
    epoch_ns("date"),
    epoch_ns("date") // N,
    "driverId" AS __parent
FROM "results"
WHERE "driverId" IN (SELECT unnest(?::BIGINT[]))
  AND (epoch_ns("date") // N IS NULL
       OR epoch_ns("date") // N <= N)
QUALIFY row_number() OVER (
    PARTITION BY "driverId"
    ORDER BY epoch_ns("date") // N DESC NULLS LAST, __row DESC
) <= N
```

### 4. Children of drivers: `standings` (7,119)

```sql
SELECT
    __row,
    "raceId", "driverId",
    "points", "position", "wins",
    epoch_ns("date"),
    epoch_ns("date") // N,
    "driverId" AS __parent
FROM "standings"
WHERE "driverId" IN (SELECT unnest(?::BIGINT[]))
  AND (epoch_ns("date") // N IS NULL
       OR epoch_ns("date") // N <= N)
QUALIFY row_number() OVER (
    PARTITION BY "driverId"
    ORDER BY epoch_ns("date") // N DESC NULLS LAST, __row DESC
) <= N
```

### 5–7. Children of drivers: `driver-dnf` train, val, test (7,119 each)

The same query runs once per split; only the table changes
(`"driver-dnf"."train"`, `"driver-dnf"."val"`, `"driver-dnf"."test"`).

```sql
SELECT
    __row,
    "driverId",
    epoch_ns("date"),
    "did_not_finish",                             -- the label
    epoch_ns("date") // N,
    "driverId" AS __parent
FROM "driver-dnf"."train"
WHERE "driverId" IN (SELECT unnest(?::BIGINT[]))
  AND (epoch_ns("date") // N IS NULL
       OR epoch_ns("date") // N <= N)             -- not after the seed's time
  AND (epoch_ns("date") // N IS NULL
       OR epoch_ns("date") // N <> N)             -- not at the target's own time
QUALIFY row_number() OVER (
    PARTITION BY "driverId"
    ORDER BY epoch_ns("date") // N DESC NULLS LAST, __row DESC
) <= N
```

### 8. Parents: `races` (1,930)

```sql
SELECT
    __row,
    "circuitId", "year", "round", "name",
    epoch_ns("date"),
    "time",
    epoch_ns("date") // N
FROM "races"
WHERE __row IN (SELECT unnest(?::BIGINT[]))
  AND (epoch_ns("date") // N IS NULL
       OR epoch_ns("date") // N <= N)
```

### 9. Parents: `constructors` (1,886)

```sql
SELECT
    __row,
    "constructorRef", "name", "nationality",
    NULL::BIGINT
FROM "constructors"
WHERE __row IN (SELECT unnest(?::BIGINT[]))
  AND TRUE
```

### 10. Children of constructors: `constructor_results` (1,093)

```sql
SELECT
    __row,
    "raceId", "constructorId",
    "points",
    epoch_ns("date"),
    epoch_ns("date") // N,
    "constructorId" AS __parent
FROM "constructor_results"
WHERE "constructorId" IN (SELECT unnest(?::BIGINT[]))
  AND (epoch_ns("date") // N IS NULL
       OR epoch_ns("date") // N <= N)
QUALIFY row_number() OVER (
    PARTITION BY "constructorId"
    ORDER BY epoch_ns("date") // N DESC NULLS LAST, __row DESC
) <= N
```

### 11. Children of constructors: `constructor_standings` (1,093)

```sql
SELECT
    __row,
    "raceId", "constructorId",
    "points", "position", "wins",
    epoch_ns("date"),
    epoch_ns("date") // N,
    "constructorId" AS __parent
FROM "constructor_standings"
WHERE "constructorId" IN (SELECT unnest(?::BIGINT[]))
  AND (epoch_ns("date") // N IS NULL
       OR epoch_ns("date") // N <= N)
QUALIFY row_number() OVER (
    PARTITION BY "constructorId"
    ORDER BY epoch_ns("date") // N DESC NULLS LAST, __row DESC
) <= N
```

### 12. Children of constructors: `qualifying` (1,093)

```sql
SELECT
    __row,
    "raceId", "driverId", "constructorId",
    "number", "position",
    epoch_ns("date"),
    epoch_ns("date") // N,
    "constructorId" AS __parent
FROM "qualifying"
WHERE "constructorId" IN (SELECT unnest(?::BIGINT[]))
  AND (epoch_ns("date") // N IS NULL
       OR epoch_ns("date") // N <= N)
QUALIFY row_number() OVER (
    PARTITION BY "constructorId"
    ORDER BY epoch_ns("date") // N DESC NULLS LAST, __row DESC
) <= N
```

### 13. Children of constructors: `results` (1,093)

```sql
SELECT
    __row,
    "raceId", "driverId", "constructorId",
    "number", "grid", "position", "positionOrder", "points",
    "laps", "milliseconds", "fastestLap", "rank", "statusId",
    epoch_ns("date"),
    epoch_ns("date") // N,
    "constructorId" AS __parent
FROM "results"
WHERE "constructorId" IN (SELECT unnest(?::BIGINT[]))
  AND (epoch_ns("date") // N IS NULL
       OR epoch_ns("date") // N <= N)
QUALIFY row_number() OVER (
    PARTITION BY "constructorId"
    ORDER BY epoch_ns("date") // N DESC NULLS LAST, __row DESC
) <= N
```

### 14. Example ranking, one page (719)

```sql
SELECT split, r
FROM (
    SELECT 'train' AS split, __row AS r,
           epoch_ns("date") // N AS ts,
           ("driverId" = N) AS same               -- same driver as the target
    FROM "driver-dnf"."train"
    WHERE "did_not_finish" IS NOT NULL
      AND epoch_ns("date") // N < N               -- before the target

    UNION ALL

    SELECT 'val' AS split, __row AS r,
           epoch_ns("date") // N AS ts,
           ("driverId" = N) AS same
    FROM "driver-dnf"."val"
    WHERE "did_not_finish" IS NOT NULL
      AND epoch_ns("date") // N < N

    UNION ALL

    SELECT 'test' AS split, __row AS r,
           epoch_ns("date") // N AS ts,
           ("driverId" = N) AS same
    FROM "driver-dnf"."test"
    WHERE "did_not_finish" IS NOT NULL
      AND epoch_ns("date") // N < N
      AND __row <> N                              -- not the target itself
)
ORDER BY same DESC, ts DESC NULLS LAST, split, r
LIMIT N OFFSET N                                  -- 64 seeds per page
```

### 15. Example fetch: `driver-dnf/val` (542)

```sql
SELECT
    __row,
    "driverId",
    epoch_ns("date"),
    "did_not_finish",
    epoch_ns("date") // N
FROM "driver-dnf"."val"
WHERE __row IN (SELECT unnest(?::BIGINT[]))       -- seeds picked by the ranking page
```

## Findings

- **`drivers` parent lookups are more than 5 times any other query.**
  `_fetch_parents` queries every parent of every row on a BFS level, including
  drivers already in the context. Example seeds are mostly the target driver's
  own rows, so most of these lookups likely return a driver already fetched.
  Skipping keys already in `visited` before querying would remove most of them.
- **Expanding a driver takes 6 children queries:** 3 database tables, plus the 3
  `driver-dnf` splits, one table each. A single `UNION ALL` over the splits would
  make it 4.
- **Each driver is expanded once:** all six children queries for drivers have
  the same count, 7,119.
- **Each ranking page re-sorts every split.** The `OFFSET` in query 14 means a
  context that needs several pages sorts the whole task table each time.
