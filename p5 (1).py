import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# Load CSV
df = pd.read_csv(r"C:\Users\tthul\Downloads\loan_approval(1).csv")

print("Dataset:")
print(df)

X = df[['Income', 'Credit_Score', 'Existing_Loans']]
y = df['Approved']

model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X, y)

income = int(input("Enter Monthly Income: "))

credit = int(input("Enter Credit Score (0-900): "))

while credit < 0 or credit > 900:
    print("Credit Score should be between 0 and 900.")
    credit = int(input("Enter Credit Score Again: "))

loan = int(input("Enter Existing Loans: "))

prediction = model.predict([[income, credit, loan]])

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")

print("\nDecision Rules:\n")
print(export_text(model, feature_names=list(X.columns)))