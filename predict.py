import joblib
model=joblib.load("house_price_model.joblib")
area=float(input("Area: ")); bedrooms=int(input("Bedrooms: "))
bathrooms=int(input("Bathrooms: ")); age=float(input("Age: "))
print("Predicted price:",model.predict([[area,bedrooms,bathrooms,age]])[0])
