import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Welcome to my UFC fighter comparer and stat tracker")

# Load the dataset
df = pd.read_csv('/Users/adamavdic/Desktop/UFC-fight-predictor/Aggregated_UFC_Fighter_Stats.csv')

# List of available metrics for comparison
available_metrics = ['SIG_STR_ratio', 'SIG_STR_pct', 'TOTAL_STR_ratio', 'TD_ratio', 'TD_pct', 'SUB.ATT',
                     'REV.', 'CTRL_seconds', 'HEAD_ratio', 'BODY_ratio', 'LEG_ratio', 'DISTANCE_ratio',
                     'CLINCH_ratio', 'GROUND_ratio']

# Function to predict the winner based on the selected metrics of two fighters
# Function to predict the winner based on the selected metrics of two fighters
def predict_winner(fighter1, fighter2, selected_metrics, aggregated_stats):
    # Retrieve the stats for both fighters
    stats1 = aggregated_stats[aggregated_stats['FIGHTER'] == fighter1]
    stats2 = aggregated_stats[aggregated_stats['FIGHTER'] == fighter2]

    # Check if both fighters are in the dataset
    if stats1.empty or stats2.empty:
        return f"One or both fighters ('{fighter1}', '{fighter2}') not found in the dataset."

    # Extract the relevant statistics for comparison
    stats1 = stats1.iloc[0]
    stats2 = stats2.iloc[0]

    # Initialize scores for each fighter
    score1 = 0
    score2 = 0

    # Lists to keep track of categories where each fighter has an advantage
    fighter1_advantages = []
    fighter2_advantages = []

    # Compare each selected metric and assign points to the fighter with better stats
    for metric in selected_metrics:
        if stats1[metric] > stats2[metric]:
            score1 += 1
            fighter1_advantages.append(metric)
        elif stats2[metric] > stats1[metric]:
            score2 += 1
            fighter2_advantages.append(metric)

    # Determine the winner based on the scores
    if score1 > score2:
        return f"{fighter1} is predicted to win based on the comparison of the selected metrics."
    elif score2 > score1:
        return f"{fighter2} is predicted to win based on the comparison of the selected metrics."
    else:
        # If the score is tied, list the categories where each fighter has the advantage
        result_message = "The fight is too close to call based on the selected metrics. Here's the breakdown:\n"
        result_message += f"{fighter1} has an advantage in: {', '.join(fighter1_advantages) if fighter1_advantages else 'None'}\n"
        result_message += f"{fighter2} has an advantage in: {', '.join(fighter2_advantages) if fighter2_advantages else 'None'}\n"
        return result_message

# Function to predict the fight result using a machine learning model
def ml_predict_winner(fighter1, fighter2, aggregated_stats):
    # Prepare the dataset for ML prediction
    features = available_metrics
    X = aggregated_stats[features]
    y = aggregated_stats['FIGHTER'].apply(lambda x: 1 if x == fighter1 else (0 if x == fighter2 else -1))
    X = X[y != -1]
    y = y[y != -1]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model's performance on the test set
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

    # Predict the fight result using the model
    fighter1_stats = aggregated_stats[aggregated_stats['FIGHTER'] == fighter1][features]
    fighter2_stats = aggregated_stats[aggregated_stats['FIGHTER'] == fighter2][features]

    # Combine the stats into a single feature vector and convert to DataFrame with appropriate column names
    combined_stats = (fighter1_stats.values + fighter2_stats.values) / 2
    combined_stats_df = pd.DataFrame(combined_stats, columns=features)

    # Use the scaler to transform the combined stats and make a prediction
    combined_stats_scaled = scaler.transform(combined_stats_df)
    prediction = model.predict(combined_stats_scaled)

    return f"The ML model predicts {'Fighter 1' if prediction[0] == 1 else 'Fighter 2'} as the winner."

# Main menu to choose prediction method
print("Choose a prediction method:")
print("1. Manual comparison of specific metrics")
print("2. Use Machine Learning model to predict the result")

choice = input("Enter your choice (1 or 2): ")

if choice == '1':
    # Get user input for fighter names
    fighter1 = input("Enter the name of the first fighter: ")
    fighter2 = input("Enter the name of the second fighter: ")

    # Display available metrics to the user
    print("Available metrics for comparison:")
    print(", ".join(available_metrics))
    print("Type 'All stats' to use all metrics for prediction.")

    # Get user input for the metrics they want to use for prediction
    selected_metrics_input = input("Enter the metrics you want to use for comparison, separated by commas (or type 'All stats'): ")

    # Check if the user chose to use all metrics or specific ones
    if selected_metrics_input.lower() == 'all stats':
        selected_metrics = available_metrics  # Use all metrics
    else:
        selected_metrics = [metric.strip() for metric in selected_metrics_input.split(',') if metric.strip() in available_metrics]

    # Check if any valid metrics were selected
    if not selected_metrics:
        print("No valid metrics selected. Please choose from the available metrics.")
    else:
        # Call the function with user input and print the result
        print(predict_winner(fighter1, fighter2, selected_metrics, df))

elif choice == '2':
    # Get user input for fighter names
    fighter1 = input("Enter the name of the first fighter: ")
    fighter2 = input("Enter the name of the second fighter: ")

    # Call the ML function and print the result
    print(ml_predict_winner(fighter1, fighter2, df))

else:
    print("Invalid choice. Please enter either 1 or 2.")
