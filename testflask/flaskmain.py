from flask import Flask, jsonify
from model import getvoxel

app = Flask(__name__)

@app.route('/get_voxel_data/<int:index>')
def get_voxel_data(index):
    data = getvoxel(index)
    return jsonify(data)

if __name__ == '__main__':
    app.run()