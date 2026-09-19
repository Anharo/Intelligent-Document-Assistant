"""
Generates a synthetic campus student-records dataset for the IDA project.
Produces 6 CSVs under data/raw/structured/:
  students.csv, attendance.csv, grades.csv, fees.csv, hostel.csv, disciplinary_records.csv

Run: uv run python scripts/generate_records.py
"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

OUT_DIR = Path("data/raw/structured")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STUDENTS = 550

PROGRAMS = [
    "B.Tech Computer Science", "B.Tech Electronics", "B.Tech Mechanical",
    "B.Sc Physics", "B.Sc Mathematics", "BBA", "B.Com", "BA English",
]
YEARS = [1, 2, 3, 4]
ENROLLMENT_STATUS = ["active", "active", "active", "active", "leave_of_absence", "withdrawn"]

COURSE_CATALOG = {
    "B.Tech Computer Science": ["CS101", "CS201", "CS301", "CS401", "MA101"],
    "B.Tech Electronics": ["EC101", "EC201", "EC301", "MA101"],
    "B.Tech Mechanical": ["ME101", "ME201", "ME301", "MA101"],
    "B.Sc Physics": ["PH101", "PH201", "MA101"],
    "B.Sc Mathematics": ["MA101", "MA201", "MA301"],
    "BBA": ["BB101", "BB201", "EC001"],
    "B.Com": ["CM101", "CM201", "EC001"],
    "BA English": ["EN101", "EN201"],
}

GRADES = ["A+", "A", "B+", "B", "C+", "C", "D", "F"]
GRADE_POINTS = {"A+": 10, "A": 9, "B+": 8, "B": 7, "C+": 6, "C": 5, "D": 4, "F": 0}
SEMESTERS = ["2024-Fall", "2025-Spring", "2025-Fall"]
MONTHS = ["2025-07", "2025-08", "2025-09"]

DISCIPLINARY_CATEGORIES = {
    "Attendance Shortage": "Student fell below minimum attendance requirement",
    "Academic Dishonesty": "Cheating/plagiarism reported during examination",
    "Hostel Rule Violation": "Violation of hostel curfew/visitor policy",
    "Alcohol Policy Violation": "Possession/consumption on campus premises",
    "Property Damage": "Damage to university property reported",
    "Ragging/Hazing": "Reported incident of hazing a junior student",
}
SANCTIONS = ["Warning", "Written Warning", "Fine Imposed", "Probation", "Suspended (1 week)", "Referred to Dean"]

HOSTEL_BLOCKS = ["A", "B", "C", "D"]

students, attendance, grades, fees, hostel, disciplinary = [], [], [], [], [], []

for i in range(1, N_STUDENTS + 1):
    sid = f"S{i:04d}"
    program = random.choice(PROGRAMS)
    year = random.choice(YEARS)
    status = random.choices(ENROLLMENT_STATUS, weights=[30, 30, 20, 15, 4, 1])[0]
    students.append({
        "student_id": sid,
        "name": fake.name(),
        "program": program,
        "year": year,
        "enrollment_status": status,
        "email": f"{sid.lower()}@campus.edu",
    })

    courses = COURSE_CATALOG[program]

    # Attendance: per course per month
    for course in courses:
        for month in MONTHS:
            held = random.randint(14, 22)
            # bias some students toward chronic low attendance
            attend_rate = random.choice([random.uniform(0.55, 0.7), random.uniform(0.75, 1.0)])
            attended = round(held * attend_rate)
            attendance.append({
                "student_id": sid, "course_code": course, "month": month,
                "classes_held": held, "classes_attended": min(attended, held),
            })

    # Grades: per course per semester (not every course every semester)
    for course in courses:
        for sem in random.sample(SEMESTERS, k=random.randint(1, len(SEMESTERS))):
            grade = random.choices(GRADES, weights=[10, 20, 20, 20, 12, 10, 5, 3])[0]
            grades.append({
                "student_id": sid, "course_code": course, "semester": sem,
                "credits": random.choice([3, 4]), "grade": grade,
                "grade_points": GRADE_POINTS[grade],
            })

    # Fees: per semester
    for sem in SEMESTERS:
        amount_due = random.choice([45000, 52000, 60000, 75000])
        paid_fraction = random.choices([1.0, 0.5, 0.0], weights=[75, 15, 10])[0]
        amount_paid = round(amount_due * paid_fraction)
        due_date = date(2025, 1, 15) if "Spring" in sem else date(2025, 8, 1)
        fees.append({
            "student_id": sid, "semester": sem, "amount_due": amount_due,
            "amount_paid": amount_paid, "due_date": due_date.isoformat(),
            "status": "paid" if amount_paid >= amount_due else ("partial" if amount_paid > 0 else "unpaid"),
        })

    # Hostel: ~60% of students live on campus
    if random.random() < 0.6:
        allotted = date(2024, 7, random.randint(1, 28))
        hostel.append({
            "student_id": sid, "block": random.choice(HOSTEL_BLOCKS),
            "room_no": f"{random.randint(1,4)}{random.randint(1,30):02d}",
            "allotted_date": allotted.isoformat(),
            "status": random.choices(["active", "vacated"], weights=[85, 15])[0],
        })

    # Disciplinary: ~12% of students have at least one record
    if random.random() < 0.12:
        for _ in range(random.randint(1, 2)):
            cat = random.choice(list(DISCIPLINARY_CATEGORIES.keys()))
            incident_date = fake.date_between(start_date=date(2024, 8, 1), end_date=date(2025, 9, 1))
            disciplinary.append({
                "student_id": sid, "incident_date": incident_date.isoformat(),
                "category": cat, "description": DISCIPLINARY_CATEGORIES[cat],
                "sanction": random.choice(SANCTIONS),
            })


def write_csv(filename, rows):
    if not rows:
        return
    path = OUT_DIR / filename
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):>6} rows -> {path}")


write_csv("students.csv", students)
write_csv("attendance.csv", attendance)
write_csv("grades.csv", grades)
write_csv("fees.csv", fees)
write_csv("hostel.csv", hostel)
write_csv("disciplinary_records.csv", disciplinary)

print("\nDone. Total students:", N_STUDENTS)