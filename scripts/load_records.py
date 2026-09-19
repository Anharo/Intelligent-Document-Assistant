"""
Creates the Postgres schema and loads the synthetic CSVs into it.

Run: uv run python scripts/load_records.py
"""
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

SCHEMA_PATH = Path("scripts/schema.sql")
DATA_DIR = Path("data/raw/structured")

TABLES_IN_ORDER = ["students", "attendance", "grades", "fees", "hostel", "disciplinary_records"]

COLUMNS = {
    "students": ["student_id", "name", "program", "year", "enrollment_status", "email"],
    "attendance": ["student_id", "course_code", "month", "classes_held", "classes_attended"],
    "grades": ["student_id", "course_code", "semester", "credits", "grade", "grade_points"],
    "fees": ["student_id", "semester", "amount_due", "amount_paid", "due_date", "status"],
    "hostel": ["student_id", "block", "room_no", "allotted_date", "status"],
    "disciplinary_records": ["student_id", "incident_date", "category", "description", "sanction"],
}


def main():
    dsn = os.getenv("POSTGRES_DSN")
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            print("Applying schema...")
            cur.execute(SCHEMA_PATH.read_text())
            conn.commit()

            for table in TABLES_IN_ORDER:
                csv_path = DATA_DIR / f"{table}.csv"
                cols = ", ".join(COLUMNS[table])
                with open(csv_path, "r") as f:
                    with cur.copy(
                        f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, HEADER true)"
                    ) as copy:
                        while data := f.read(8192):
                            copy.write(data)
                conn.commit()

                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                print(f"Loaded {table:<22} {count:>6} rows")

    print("\nDone.")


if __name__ == "__main__":
    main()
