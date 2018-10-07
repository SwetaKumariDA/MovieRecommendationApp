from flask import Flask, render_template, request, Response
from MovieRecommendationEngine import *
import json
app = Flask(__name__)

NAMES=''
x=''

@app.route('/autocomplete',methods=['GET'])
def autocomplete():
    return Response(json.dumps(NAMES), mimetype='application/json')

@app.route('/',methods=['GET','POST'])
def index():
    return render_template("index.html")

@app.route("/getreco", methods=['GET', 'POST'])
def getMovieReco():
    inp_movie=''
    if request.form:
       inp_movie = request.form['fname']
    val = x.movies_recommendations(inp_movie)
    return render_template("index.html", out=val, rec="Movie recommendations for you")

if __name__ == '__main__':
    x = MovieRecommendation()
    NAMES=x.getmovielist()
    app.run()
