import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.impute import KNNImputer

# Load and clean the dataset
df = pd.read_csv('/Users/adamavdic/Desktop/UFC-fight-predictor/enhanced_fighter_stats.csv')

# Impute missing values using KNN Imputer for better imputations
imputer = KNNImputer(n_neighbors=5)
df[['win_loss_ratio']] = imputer.fit_transform(df[['win_loss_ratio']])

# Drop any remaining rows with missing values across all relevant columns to ensure data consistency
df.dropna(subset=['TD', 'BODY', 'LEG', 'CLINCH', 'GROUND', 'win_loss_ratio'], inplace=True)

# Categorize win-loss ratio into classes
# "Low" < 0.4, "Medium" between 0.4 and 0.7, "High" > 0.7
def categorize_win_loss_ratio(ratio):
    if ratio < 0.4:
        return 'Low'
    elif 0.4 <= ratio <= 0.7:
        return 'Medium'
    else:
        return 'High'

df['win_loss_category'] = df['win_loss_ratio'].apply(categorize_win_loss_ratio)

# Feature Engineering: Add interaction terms and create new features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interaction_features = poly.fit_transform(df[['TD', 'BODY', 'LEG', 'CLINCH', 'GROUND']])
interaction_feature_names = poly.get_feature_names_out(['TD', 'BODY', 'LEG', 'CLINCH', 'GROUND'])
interaction_df = pd.DataFrame(interaction_features, columns=interaction_feature_names)

# Adding new features that could impact win-loss ratio
df['SIG_STR_TOTAL'] = df['SIG.STR.'] * df['TOTAL STR.']  # Corrected column names
df['recent_win_rate'] = np.random.uniform(0.5, 1, size=len(df))  # Placeholder for actual recent win rate

# Combine features
X = pd.concat([df[['TD', 'BODY', 'LEG', 'CLINCH', 'GROUND', 'SIG_STR_TOTAL', 'recent_win_rate']], interaction_df], axis=1)
y = df['win_loss_category']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features to improve model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models for ensemble classification
xgb_model = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.01, subsample=0.6, colsample_bytree=1.0, random_state=42)
catboost_model = CatBoostClassifier(iterations=150, learning_rate=0.05, depth=5, verbose=0)
gbm_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)

# Ensemble - StackingClassifier
stacked_model = StackingClassifier(estimators=[('xgb', xgb_model), ('catboost', catboost_model), ('gbm', gbm_model)], final_estimator=GradientBoostingClassifier())

# Train the stacked model
stacked_model.fit(X_train_scaled, y_train)

# Main menu to choose functionality
print("Welcome to the UFC Fight Predictor!")
print("Choose an option:")
print("1. Predict win-loss category using Machine Learning model")
print("2. Compare two fighters using rules-based comparison")
choice = input("Enter your choice (1 or 2): ")

if choice == '1':
    # Predict win-loss category for a fighter using the model
    fighter_name = input("Enter the name of the fighter to predict the win-loss category: ")

    def predict_win_loss_category_from_csv(fighter_name, df):
        # Check if the fighter exists in the dataset
        if fighter_name not in df['FIGHTER'].values:
            return f"Fighter {fighter_name} not found in the dataset."

        # Retrieve the fighter's stats from the dataset
        fighter_stats = df[df['FIGHTER'] == fighter_name][['TD', 'BODY', 'LEG', 'CLINCH', 'GROUND']].values[0]

        # Add interaction features for the fighter's stats
        interaction_stats = poly.transform([fighter_stats])[0]

        # Combine original stats with the interaction features and add new features like SIG_STR_TOTAL and recent_win_rate
        sig_str_total = df[df['FIGHTER'] == fighter_name]['SIG.STR.'].values[0] * df[df['FIGHTER'] == fighter_name]['TOTAL STR.'].values[0]
        recent_win_rate = np.random.uniform(0.5, 1)  # Placeholder for actual recent win rate if available

        # Create the final feature vector
        fighter_stats_combined = np.concatenate((fighter_stats, [sig_str_total, recent_win_rate], interaction_stats))

        # Scale the fighter's stats using the same scaler as the model
        fighter_stats_scaled = scaler.transform([fighter_stats_combined])
        
        # Predict the win-loss category using the trained model
        predicted_category = stacked_model.predict(fighter_stats_scaled)
        return f"Predicted win-loss category for {fighter_name}: {predicted_category[0]}"

    # Predict the win-loss category for the selected fighter using stats from the CSV
    predicted_category = predict_win_loss_category_from_csv(fighter_name, df)
    print(predicted_category)

elif choice == '2':
    # Compare two fighters using the rules-based approach
    fighter1 = input("Enter the name of the first fighter: ")
    fighter2 = input("Enter the name of the second fighter: ")

    # Function to compare two fighters using the aggregated statistics provided
    def rules_based_comparison_aggregated(fighter1, fighter2, aggregated_stats):
        # Retrieve aggregated stats for each fighter
        remaining_features = ['TD', 'BODY', 'LEG', 'CLINCH', 'GROUND', 'win_loss_ratio']
        fighter1_stats = aggregated_stats[aggregated_stats['FIGHTER'] == fighter1][remaining_features].mean()
        fighter2_stats = aggregated_stats[aggregated_stats['FIGHTER'] == fighter2][remaining_features].mean()

        # Ensure there is data for both fighters
        if fighter1_stats.empty or fighter2_stats.empty:
            return "One or both fighters not found in the dataset."

        # Initialize scores for each fighter
        fighter1_score = 0
        fighter2_score = 0

        # List to keep track of the categories where each fighter has an advantage
        fighter1_advantages = []
        fighter2_advantages = []

        # Detailed results string to show all metric comparisons
        result_details = f"Comparison of aggregated metrics for {fighter1} and {fighter2}:\n"

        # Compare each metric, display the values, and determine which fighter has the advantage
        for metric in remaining_features:
            result_details += f"{metric}: {fighter1} = {fighter1_stats[metric]:.4f}, {fighter2} = {fighter2_stats[metric]:.4f} -> "

            if fighter1_stats[metric] > fighter2_stats[metric]:
                fighter1_score += 1
                fighter1_advantages.append(metric)
                result_details += f"{fighter1} has the advantage\n"
            elif fighter2_stats[metric] > fighter1_stats[metric]:
                fighter2_score += 1
                fighter2_advantages.append(metric)
                result_details += f"{fighter2} has the advantage\n"
            else:
                result_details += "No advantage\n"

        # Determine the winner based on the scores
        if fighter1_score > fighter2_score:
            result = f"{fighter1} is predicted to win based on the comparison of the key metrics."
        elif fighter2_score > fighter1_score:
            result = f"{fighter2} is predicted to win based on the comparison of the key metrics."
        else:
            result = "The fight is too close to call based on the selected metrics."

        # Add details about the advantages each fighter has
        result += f"\n\n{fighter1} has an advantage in: {', '.join(fighter1_advantages) if fighter1_advantages else 'None'}"
        result += f"\n{fighter2} has an advantage in: {', '.join(fighter2_advantages) if fighter2_advantages else 'None'}"

        return result_details + "\n" + result

    # Run the rules-based comparison with the aggregated statistics for the selected fighters
    aggregated_result = rules_based_comparison_aggregated(fighter1, fighter2, df)

    # Display the full result of the rules-based comparison using aggregated statistics
    print(aggregated_result)
    