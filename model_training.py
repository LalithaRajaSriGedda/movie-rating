from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_and_evaluate(df):
    features = ['Name','Year','Duration','Genre','Rating','Votes', 'Director', 'director_success_rate', 'genre_avg_rating']
    X = df[features]
    y = df['Rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    joblib.dump(model, r"model.pkl.py")
    return mse, r2
