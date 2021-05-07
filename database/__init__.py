import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

db_connection = psycopg2.connect(
    database=os.environ["DATABASE"],
    user=os.environ["USER"],
    password=os.environ["PASSWORD"],
    host=os.environ["HOST"],
    port="5432",
)

cur = db_connection.cursor()

cur.execute(
    """
CREATE TABLE history (
    id serial PRIMARY KEY,
    date date,
    features text,
    prediction numeric 
);
"""
)

db_connection.commit()
db_connection.close()
