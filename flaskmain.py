from flask import Flask, jsonify
# import os
# import sys
# o_path = os.getcwd()
# sys.path.append(o_path)
# from export_post_clip import export_voxel

app = Flask(__name__)

@app.route('/get_embeddings/<int:index>')
def get_voxel_data(index):
    data = [1232]
    return jsonify(data)

if __name__ == '__main__':
    # a = [12,4,5,26,1,4]
    # print (a)
    app.run()