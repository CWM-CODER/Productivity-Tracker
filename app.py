import subprocess
import sys
from flask import Flask,render_template,request,redirect,url_for
import sqlite3

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')




@app.route('/log_form',methods=['GET','POST'])
def log_form():
    if request.method == 'POST':
        date = request.form['date']
        activity = request.form['activity']
        duration = request.form['duration']
        description = request.form.get('description','')


        # Save to database
        conn = sqlite3.connect('db/tracker.db')
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO logs (date,activity,duration,description)
        VALUES (?,?,?,?)

    ''',(date,activity,duration,description))
        conn.commit()
        conn.close()

        return redirect("http://localhost:8501")

    return render_template('log_form.html')

def start_streamlit():
    subprocess.Popen(
        ["streamlit", "run", "dashboard.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


if __name__ == "__main__":
    start_streamlit()
    app.run(debug = True,use_reloader = False)

    
