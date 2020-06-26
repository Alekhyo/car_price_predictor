from flask import Flask,render_template,request
import pickle
import pandas as pd

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
data=pd.read_csv('quickr.csv')

@app.route('/')
def home():
    return render_template('home.html',data=data)


@app.route('/predict',methods=['POST'])
def predict():
    try:
        company=request.form.get('company')
        year=request.form.get('year')
        fuel=request.form.get('fuel')
        distance=request.form.get('distance')

        df = pd.DataFrame(columns=['distance', 'year', 'BMW', 'Chevrolet', 'Datsun', 'Fiat', 'Ford',
                                   'Honda', 'Hyundai', 'Jeep', 'Mahindra', 'Maruti',
                                   'Mercedes', 'Mitsubishi', 'Nissan', 'Premier', 'Renault',
                                   'Skoda', 'Ssangyong', 'Tata', 'Toyota', 'Volkswagen', 'Volvo', 'Diesel',
                                   'Electric', 'Hybrid', 'LPG', 'Petrol'])

        d={'company':company,'distance':distance,'year':year,'fuel':fuel}
        dp=pd.DataFrame(d,index=[1])



        F=pd.get_dummies(dp,columns=['company','fuel'])

        col = F.columns.tolist()
        col[2] = col[2].split("_")[1]
        col[3] = col[3].split("_")[1]
        F.columns = col

        df[F.columns]=F[F.columns]

        df.fillna(0,inplace=True)

        predict=int(round(model.predict(df)[0],0))

        return render_template('home.html',predict=predict,label=1)

    except:
        return render_template('home.html',label=0)



if __name__ =="__main__":
    app.run(debug=True)