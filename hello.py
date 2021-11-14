from flask import Flask, render_template, request

app = Flask(__name__)
list = []


@app.route('/', methods=['POST', 'GET'])
def home():
    print(request)
    return render_template('home.html')

@app.route('/post',methods=['POST', 'GET'] )
def post():
    result=request.form
    print(result.to_dict(flat=False))
    fn=result.get('firstname')
    ln=result.get('lastname')
    rs='hey'+" "+(fn+ln)
    return render_template('post.html', somme=rs)

if __name__ == "__main__":
    app.run(debug=True)
