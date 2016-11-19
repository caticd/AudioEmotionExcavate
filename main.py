from flask import Flask, render_template, request, redirect, url_for
from flask_bootstrap import Bootstrap
import os
from models import SqlSession, Files

app = Flask(__name__)
bootstrap = Bootstrap(app)

app.config['UPLOAD_DIR'] = "C:\Users\ZYC\PycharmProjects\AudioEmotionExcavate\upload_dir"


@app.route('/')
def index():
    sql_session = SqlSession()
    page = request.args.get('page') or 1
    count = sql_session.query(Files).count()
    total_page = count / 25
    if count % 25:
        total_page += 1
    items = sql_session.query(Files).order_by(Files.id.desc()).offset((int(page) - 1) * 25).limit(25).all()
    return render_template('index.html', items=items, cur_page=page, total_page=total_page)


@app.route('/upload_file/', methods=['POST', 'GET'])
def upload():
    sql_session = SqlSession()
    if 'file[]' in request.files:
        upload_files = request.files.getlist("file[]")
        for upload_file in upload_files:
            if upload_file:
                li = sql_session.query(Files).count()
                if li > 0:
                    count = sql_session.query(Files).order_by(Files.id.desc()).first().id
                else:
                    count = 0
                filename = str(count + 1) + ".wav"
                upload_file.save(os.path.join(app.config['UPLOAD_DIR'], filename))
                the_file = Files(upload_file.filename, filename, 1)
                sql_session.add(the_file)
        sql_session.commit()
    return redirect(url_for("index"))


if __name__ == '__main__':
    app.run()
