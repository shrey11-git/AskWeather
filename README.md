# 🌤️ AskWeather (Weather predictor & Classifier)
A user-friendly weather classifier powered by machine learning and wrapped in a slick GUI.

# ※ What Problem Does It Solve?
A machine learning-based weather classification tool that predicts daily weather types using temperature and precipitation inputs. Includes a user-friendly GUI for quick, code-free predictions.

# ※ How It Came to Life
We often rely on news apps or vague “chance of rain” forecasts. But what if we could learn from historical weather patterns to build our own intuitive weather labeling tool?

AskWeather is a hands-on project that blends classic machine learning with real meteorological data — in this case, over 30 years of data from Mumbai (Santacruz).

We trained a classifier that predicts whether a day is sunny, rainy, snowy, humid, cloudy, or cold — based on just a few values you can get from any weather API or sensor.

# ※ Key Features
- CSV Input: Reads decades of weather data (Mumbai_1990_2022_Santacruz.csv).
- Preprocessing:
Cleans missing data, 
Calculates temp_range, 
Creates labels like sunny, rainy, etc.
- Model Training:
Uses a Random Forest Classifier, 
Evaluated with test data and cross-validation
- Prediction Function:
Easily reusable with new input values
- GUI App (Tkinter):
Enter temperature and precipitation, 
Click a button to get instant weather prediction
- Unit Tested:
Test suite with edge cases and validation

# ※ Tech Stack
- Language- Python 3.9+
- ML & Evaluation- scikit-learn
- Data Handling- pandas
- GUI- tkinter, Pillow
- Testing- unittest
- Visualization- PIL (image handling)

# ※ How to Run
- Clone the repo
git clone https://github.com/shrey11-git/AskWeather.git
cd askweather
- Install dependencies
pip install -r requirements.txt
- Run the main app
python main.py
- This will:
Train the model on historical data, 
Evaluate it, 
Launch the GUI interface for weather prediction

# ※ Screenshot
<img width="391" height="681" alt="AskWeather" src="https://github.com/user-attachments/assets/2ff91a88-25ea-4f13-9693-5067a5841ef6" />

# ※ What I Learned
- Cleaning and labeling raw real-world climate data
- Defining logical thresholds for weather types
- Training robust models (Random Forest)
- Building smooth user interfaces with Tkinter
- Writing maintainable, testable ML code

# ※ Author
Shreyash Tiwari
[ GitHub](https://github.com/shrey11-git) • [linkedin](http://www.linkedin.com/in/shreyashtiwari-csbs)

TL;DR Summary: 
AskWeather is a weather label predictor trained on historical climate data.
Just enter temp and rainfall values — and get an instant weather forecast using a friendly GUI.
It’s fast, accurate, and built to teach you the power of small data + smart models.
