from flask import Flask, jsonify
import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
from test_post_clip_test import export_voxel

app = Flask(__name__)

@app.route('/get_voxel_data/<int:index>')
def get_voxel_data(index):
    data = export_voxel()
    return jsonify(data)

if __name__ == '__main__':
    app.run()