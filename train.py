import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# CSV must contain: area, bedrooms, bathrooms, age, price
df=pd.read_csv("houses.csv")
X=df[["area","bedrooms","bathrooms","age"]]; y=df["price"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
model=RandomForestRegressor(n_estimators=200,random_state=42)
model.fit(Xtr,ytr)
print("MAE:",mean_absolute_error(yte,model.predict(Xte)))
joblib.dump(model,"house_price_model.joblib")
print("Saved model.")
