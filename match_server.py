import os
from flask import Flask, render_template, request, url_for, send_from_directory
import json
import random


# from flask_wtf.file import FileField, FileRequired, FileAllowed
# class UploadForm(FlaskForm):
#     photo = FileField('Upload Image', validators=[FileRequired(), FileAllowed(['jpg','jpeg','png','gif'])])
#     submit = SubmitField()

app = Flask(__name__)
# UPLOAD_FOLDER = 'request_upload'
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

match_file = "match_result.txt"
match_map = {}

with open(os.path.join(BASE_DIR, match_file), 'r') as f:
    for line in f.readlines():
        data = json.loads(line)
        match_map[data["query_skuId"]] = data["suits"]



@app.route('/match', methods=['POST'])
def Hello_cloth():
    output_str = ""
    output_json = {}
    status = 'ok'
    if request.method == 'POST':
        input_str = json.loads(request.data)
        if "skuId" not in input_str.keys():
            return "please use true params"
        skuId = input_str["skuId"]
        if skuId not in match_map.keys():
            return "%s not in match list" % skuId
        matchs = match_map[skuId]
        tmp = random.randint(0, len(matchs) - 1)
        items = {}
        for value in matchs[tmp]:
            items[value["skuId"]] = value["url"]
        return json.dumps(items)
    return "please use POST request !"




if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5001)
