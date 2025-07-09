import pandas as pd

# Part 1- Code to Load & Explore CSV
# Load the csv dataset
df = pd.read_csv("Mumbai_1990_2022_Santacruz.csv")

# show initial rows
print(df.head())

# null values check
print("\nMissing values:\n", df.isna().sum())

# basic statistics check
print("\nDataset summary:\n", df.describe())

# column names check
print("\nColumns in dataset:", df.columns.tolist())

# part 2- Preprocess & Add Labels
# Drop rows having missing data
df = df.dropna()

# temperature range feature
df['temp_range'] = df['tmax'] - df['tmin']

# labels based on thresholds
def classify_weather(row):
    if row['tavg'] < 2 and row['prcp'] > 0:
        return 'snowy'
    if row['prcp'] > 20:
        return 'rainy'
    elif 15 <= row['tavg'] <= 22 and 5 <= row['prcp'] <= 15:
        return 'cloudy'
    elif row['tavg'] < 22:
        return 'cold'
    elif row['tavg'] > 30 and row['prcp'] < 5:
        return 'sunny'
    elif row['prcp'] > 5 and row['tavg'] > 25:
        return 'humid'
    else:
        return 'sunny'  # fallback option

df['label'] = df.apply(classify_weather, axis=1)

# class balance check
print("\nClass distribution:\n", df['label'].value_counts())

# Final features
features = ['tavg', 'tmin', 'tmax', 'prcp', 'temp_range']
X = df[features]
y = df['label']

# part 3- Train-Test Split
from sklearn.model_selection import train_test_split

# Split data into training and test sets (80%->train, 20%-> test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

#part 4- Training the Model
from sklearn.ensemble import RandomForestClassifier

# Initialz and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model training done!!!")

#part 5- Evaluate the Model
from sklearn.metrics import classification_report, accuracy_score

# Predict on test set
y_pred = model.predict(X_test)

# performance evaluation
print("\n🔍 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Accuracy: {cv_scores.mean():.4f}")

#Prediction Function
def predict_weather(tavg, tmin, tmax, prcp):
    input_data = pd.DataFrame([{
        'tavg': tavg,
        'tmin': tmin,
        'tmax': tmax,
        'prcp': prcp
    }])
    input_data['temp_range'] = input_data['tmax'] - input_data['tmin']
    prediction = model.predict(input_data)[0]
    return prediction

#part 6- Test the Prediction Function
test_pred = predict_weather(tavg=28.5, tmin=22.0, tmax=34.2, prcp=0.0)
print(f"🔮 Predicted Weather: {test_pred}")

#import and call gui module
from gui import launch_gui

#Launch the gui
launch_gui(predict_weather)

