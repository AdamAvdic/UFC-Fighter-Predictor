# UFC-Fighter-Predictor
UFC Fighter Comparer and Stat Tracker
Welcome to the UFC Fighter Comparer and Stat Tracker! This project allows you to compare two UFC fighters based on their stats and predict the likely winner using manual comparison or a machine learning model.

Project Overview
The UFC Fighter Comparer and Stat Tracker provides two prediction methods:

Manual Comparison: Compare specific metrics between two fighters to predict the winner.
Machine Learning Model: Use a Random Forest Classifier to predict the winner based on historical fight statistics.
The project uses a dataset of aggregated UFC fighter statistics to perform these comparisons and predictions.

Features
Metric-Based Comparison: Compare fighters using key statistics like significant strike ratio, takedown percentage, and more.
Machine Learning Prediction: Uses a trained Random Forest model to predict the winner based on selected metrics.
Interactive Console: The tool offers a menu-driven interface to select the desired prediction method and enter fighter information.
Getting Started
Prerequisites
To run this project, you'll need the following installed on your system:

Python 3.x
Required Python libraries:
pandas
scikit-learn
You can install the required libraries using pip:

bash
Copy code
pip install pandas scikit-learn
Dataset
The project uses a dataset named Aggregated_UFC_Fighter_Stats.csv, which should be placed in the same directory as the main Python script. Ensure the dataset is formatted correctly with the following columns:

FIGHTER
SIG_STR_ratio
SIG_STR_pct
TOTAL_STR_ratio
TD_ratio
TD_pct
SUB.ATT
REV.
CTRL_seconds
HEAD_ratio
BODY_ratio
LEG_ratio
DISTANCE_ratio
CLINCH_ratio
GROUND_ratio
Running the Project
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/UFC-Fighter-Comparer.git
cd UFC-Fighter-Comparer
Run the main script:

bash
Copy code
python ufc_fighter_comparer.py
Follow the on-screen instructions to choose a prediction method, enter fighter names, and view the results.

Usage
Option 1: Manual Comparison
Enter the names of the two fighters.
Select specific metrics for comparison or choose to use all metrics.
The tool will compare the selected metrics and predict the winner.
Option 2: Machine Learning Prediction
Enter the names of the two fighters.
The tool will use the Random Forest Classifier to predict the winner based on historical fight statistics.
Project Structure
bash
Copy code
UFC-Fighter-Comparer/
│
├── Aggregated_UFC_Fighter_Stats.csv   # Dataset file
├── ufc_fighter_comparer.py            # Main script
└── README.md                          # Project documentation (you are here)
Future Improvements
Implement a graphical user interface (GUI) for enhanced user experience.
Incorporate additional machine learning algorithms for comparison.
Add data visualization to show fighter statistics.
Contributing
Contributions are welcome! Please feel free to submit issues or pull requests to improve the project.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
This project uses a dataset compiled from UFC fighter statistics.
Thanks to the open-source community for the tools and libraries that made this project possible.
