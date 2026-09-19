-- IDA Campus/HR records schema
-- Run against the ida_records database (see docker-compose.yml)

DROP TABLE IF EXISTS disciplinary_records CASCADE;
DROP TABLE IF EXISTS hostel CASCADE;
DROP TABLE IF EXISTS fees CASCADE;
DROP TABLE IF EXISTS grades CASCADE;
DROP TABLE IF EXISTS attendance CASCADE;
DROP TABLE IF EXISTS students CASCADE;

CREATE TABLE students (
    student_id        VARCHAR(10) PRIMARY KEY,
    name               VARCHAR(150) NOT NULL,
    program            VARCHAR(100) NOT NULL,
    year               SMALLINT NOT NULL CHECK (year BETWEEN 1 AND 4),
    enrollment_status  VARCHAR(30) NOT NULL,
    email              VARCHAR(150) UNIQUE NOT NULL
);

CREATE TABLE attendance (
    id                SERIAL PRIMARY KEY,
    student_id        VARCHAR(10) NOT NULL REFERENCES students(student_id),
    course_code       VARCHAR(10) NOT NULL,
    month             VARCHAR(7) NOT NULL,      -- 'YYYY-MM'
    classes_held      SMALLINT NOT NULL,
    classes_attended  SMALLINT NOT NULL
);

CREATE TABLE grades (
    id            SERIAL PRIMARY KEY,
    student_id    VARCHAR(10) NOT NULL REFERENCES students(student_id),
    course_code   VARCHAR(10) NOT NULL,
    semester      VARCHAR(20) NOT NULL,
    credits       SMALLINT NOT NULL,
    grade         VARCHAR(3) NOT NULL,
    grade_points  SMALLINT NOT NULL
);

CREATE TABLE fees (
    id             SERIAL PRIMARY KEY,
    student_id     VARCHAR(10) NOT NULL REFERENCES students(student_id),
    semester       VARCHAR(20) NOT NULL,
    amount_due     INTEGER NOT NULL,
    amount_paid    INTEGER NOT NULL,
    due_date       DATE NOT NULL,
    status         VARCHAR(10) NOT NULL
);

CREATE TABLE hostel (
    id             SERIAL PRIMARY KEY,
    student_id     VARCHAR(10) NOT NULL REFERENCES students(student_id),
    block          VARCHAR(5) NOT NULL,
    room_no        VARCHAR(10) NOT NULL,
    allotted_date  DATE NOT NULL,
    status         VARCHAR(10) NOT NULL
);

CREATE TABLE disciplinary_records (
    id             SERIAL PRIMARY KEY,
    student_id     VARCHAR(10) NOT NULL REFERENCES students(student_id),
    incident_date  DATE NOT NULL,
    category       VARCHAR(50) NOT NULL,
    description    TEXT NOT NULL,
    sanction       VARCHAR(50) NOT NULL
);

CREATE INDEX idx_attendance_student ON attendance(student_id);
CREATE INDEX idx_grades_student ON grades(student_id);
CREATE INDEX idx_fees_student ON fees(student_id);
CREATE INDEX idx_hostel_student ON hostel(student_id);
CREATE INDEX idx_disciplinary_student ON disciplinary_records(student_id);
