import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score

np.random.seed(42)
n_samples = 1000

# temp,vib,pre. = np.random.normal(mean ,stdDev,  n_samples)
temperature = np.random.normal(75,10,n_samples)
vibraton = np.random.normal(0.02 , 0.005 , n_samples)
pressure = np.random.normal(100,15,n_samples)

failure = (temperature > 85 ) | (vibraton > 0.03)| (pressure > 120)
failure = failure.astype(int)

data = pd.DataFrame({
    'Tempreature':temperature, 
     'Vibration':vibraton, 
     'Pressure' :pressure,
     'Failure':failure})

print(data.head())
print(data.columns)

x = data [['Tempreature', 'Vibration', 'Pressure']]
y = data ['Failure']

x_train , x_test , Y_train , Y_test = train_test_split( x , y ,test_size= 0.2 , random_state=42)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = RandomForestClassifier(n_estimators=100, random_state=42 )
model.fit(x_train , Y_train)

Y_pred = model.predict(x_test)
print("Accureacy:" , accuracy_score(Y_test ,Y_pred))
print("\nClassification Report :\n", classification_report(Y_test, Y_pred))
print("\nconfusion Mateix : \n", confusion_matrix(Y_test , Y_pred))

# def predict_mantanance(temperature , vibration, pressure, model , scaler ):
#     input_data = np.array([temperature,vibration, pressure])
#     input_data = input_data.reshape(1 , -1)
#     input_data = scaler.transform(input_data)
#     prediction = model.predict(input_data)
#     return "Faliure" if prediction[0] == 1 else "No Faliure"

# new_state = predict_mantanance(90 , 0.035 , 110 , model ,scaler )
# print(f"Predictions :- {new_state}")