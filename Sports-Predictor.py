import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Simulate sports betting dataset
def generate_sports_data(n=1000):
    np.random.seed(42)
    random.seed(42)
    
    data = {
        'team_1_strength': np.random.uniform(0, 1, n),
        'team_2_strength': np.random.uniform(0, 1, n),
        'home_advantage': np.random.choice([0, 1], n),
        'previous_win_rate_t1': np.random.uniform(0.3, 0.8, n),
        'previous_win_rate_t2': np.random.uniform(0.3, 0.8, n),
        'betting_odds_t1': np.random.uniform(1.5, 3.0, n),
        'betting_odds_t2': np.random.uniform(1.5, 3.0, n),
    }
    
    df = pd.DataFrame(data)
    df['team_1_wins'] = (df['team_1_strength'] + df['home_advantage'] * 0.1 + df['previous_win_rate_t1'] * 0.5) > (
        df['team_2_strength'] + df['previous_win_rate_t2'] * 0.5)
    df['team_1_wins'] = df['team_1_wins'].astype(int)
    
    return df

# Generate dataset
df = generate_sports_data()

# Split dataset into training and test sets
X = df.drop(columns=['team_1_wins'])
y = df['team_1_wins']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train predictive model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy:.2f}')

# Function to predict a match outcome
def predict_match(team_1_strength, team_2_strength, home_advantage, prev_win_rate_t1, prev_win_rate_t2, odds_t1, odds_t2):
    match_data = np.array([[team_1_strength, team_2_strength, home_advantage, prev_win_rate_t1, prev_win_rate_t2, odds_t1, odds_t2]])
    prediction = model.predict(match_data)
    return 'Team 1 Wins' if prediction[0] == 1 else 'Team 2 Wins'

# Example prediction
print(predict_match(0.7, 0.5, 1, 0.6, 0.4, 2.0, 2.5))
