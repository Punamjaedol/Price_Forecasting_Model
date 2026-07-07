import pyodbc as db
import sys
from config import *


def get_connection():
    """Return DB connection information"""
    return db.connect(
        f"DRIVER={{SQL Server}};"
        f"SERVER={DB_HOST},{DB_PORT};"
        f"DATABASE={DB_NAME};"
        f"UID={DB_USER};"
        f"PWD={DB_PASSWORD};"
        "TrustServerCertificate=yes;"
    )

def connect():
    """Connect to DB. Returns connection object or None if connection fails"""
    try:
        conn = get_connection()
        # cursor = conn.cursor()
        print(f"[DB][CONNECT] {DB_HOST}:{DB_PORT} -> SUCCESS")

        return conn

    except db.Error as e:
        print(f"[DB][DISCONNECT] Error connecting to DB Platform: {e}")
        sys.exit(1)
        return None
    
def close_connection(conn):
    if conn: conn.close()

# =====================
#       SQL Query
# =====================
def select(table, columns="*", where=None, params=None, extra=""):
    """
    SELECT * FROM table WHERE ...
    where example: "id = ? AND name = ?"
    params example: (1, "kim")
    """
    try:
        conn = connect()
        with conn.cursor() as cur:
            sql = f"SELECT {columns} FROM {table}"
            if where:
                sql += f" WHERE {where}"
            if extra:
                sql += f" {extra}"

            # Handle separately because pyodbc treats None as a valid second argument.
            if params is None:
                cur.execute(sql)
            else:
                cur.execute(sql, params)
            result = cur.fetchall()

            print(f"[DB][SELECT] {table} -> {len(result)} rows")
            return result

    except db.Error as e:
        print(f"[DB][SELECT][ERROR] {table} | {e}")
        return None
    finally:
        close_connection(conn)

def insert(table, data: dict):
    """
    data = {"col1": val1, "col2": val2}
    """
    try:
        conn = connect()

        cols = ",".join(data.keys())
        vals = ",".join(["?"] * len(data))

        sql = f"INSERT INTO {table} ({cols}) VALUES ({vals})"
        
        with conn.cursor() as cur:
            cur.execute(sql, tuple(data.values()))
        conn.commit()

        print(f"[DB][INSERT] {table} -> SUCCESS")

    except db.Error as e:
        print(f"[DB][INSERT][ERROR] {table} | {e}")
        sys.exit(1)
    finally:
        close_connection(conn)
