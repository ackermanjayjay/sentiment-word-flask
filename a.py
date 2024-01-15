from flask import Flask, render_template, request, url_for, flash, redirect
from transformers import pipeline

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("a.html",title="Program Analisis Sentiment")    

@app.route('/hasil', methods=('GET', 'POST'))
def hasil():
    result = []
    kalimat  = ""
    pretrained_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
    if request.method == "POST":
        getKalimat = request.form.get("kalimat")
        nlp = pipeline(
        "sentiment-analysis",
        model=pretrained_name,
        tokenizer=pretrained_name
    )        
        result = nlp(getKalimat)
       
  
    return render_template("a.html",title="Hasil Analisis Sentiment",kalimat=getKalimat,result=result) 
 
if __name__ == '__main__':
    app.run(debug=True)