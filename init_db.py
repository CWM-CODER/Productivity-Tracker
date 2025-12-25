import sqlite3
import os

os.makedirs('db',exist_ok=True)

# connect to the databse 

conn = sqlite3.connect('db/tracker.db')
cursor = conn.cursor()

## creating log table

cursor.execute('''
CREATE TABLE IF NOT EXISTS logs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,column aded
    activity TEXT NOT NULL,
    duration INTEGER NOT NULL,
    description TEXT,
    user_label INTEGER
)

''')


conn.commit()
conn.close()

print("Logs table create successfully in tracker.db")
