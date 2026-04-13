import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv("crop.csv")

df['Disease_Description'] = df['Disease_Description'].astype(str).fillna('')
df['Crop_Type'] = df['Crop_Type'].astype(str).fillna('Unknown')
df['Soil_Type'] = df['Soil_Type'].astype(str).fillna('Unknown')
df['Irrigation_Method'] = df['Irrigation_Method'].astype(str).fillna('Unknown')
df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
df['Temperature'].fillna(df['Temperature'].median(), inplace=True)

print(df)

# text_features = ['Disease_Description']
# categorical_features = ['Crop_Type', 'Soil_Type', 'Irrigation_Method']
# numerical_features = ['Temperature']

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(stop_words='english'), 'Disease_Description'),  # <- string, not list
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Crop_Type','Soil_Type','Irrigation_Method']),
        ('num', 'passthrough', ['Temperature'])
    ]
)


X = df[['Crop_Type','Soil_Type','Irrigation_Method','Temperature','Disease_Description']]
y = df['Disease_Name'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📏 Data Split: {len(X_train)} Training samples, {len(X_test)} Test samples")
print(X_test,y_test)


print("🚀 Training Model on Training Set...")
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
print("✅ Training Complete!")

y_pred = model.predict(X_test)

print("Model Performance on Unseen Test Data:")
accuracy = model.score(X_test, y_test)
print(accuracy)




