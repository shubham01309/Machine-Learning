import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

file_path = 'medicine.csv'
dataset = pd.read_csv(file_path)

label_encoder = LabelEncoder()
dataset['Reason_encoded'] = label_encoder.fit_transform(dataset['Reason'])

tfidf = TfidfVectorizer(stop_words='english', max_features=500)
description_features = tfidf.fit_transform(dataset['Description']).toarray()

X = description_features
y = dataset['Reason_encoded']
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

similarity = cosine_similarity(description_features)

def recommend_top5_medicines(input_reason):
    input_features = tfidf.transform([input_reason]).toarray()
    
    predicted_label = knn.predict(input_features)[0]
    
    matching_medicines = dataset[dataset['Reason_encoded'] == predicted_label]['Drug_Name'].value_counts()
    top_5_medicines = matching_medicines.head(5).index.tolist()
    
    return label_encoder.inverse_transform([predicted_label])[0], top_5_medicines

def recommend_similar_medicines(medicine):
    
    medicine_index = dataset[dataset['Drug_Name'] == medicine].index[0]

    distances = similarity[medicine_index]

    medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    similar_medicines = [dataset.iloc[i[0]].Drug_Name for i in medicines_list]
    return similar_medicines

st.title("Personalized Medicine Recommendation System")

st.header("Find Top 5 Recommended Medicines")
all_reasons = sorted(dataset['Reason'].unique())
input_reason = st.selectbox("Select a medical condition:", options=all_reasons)

if st.button("Recommend Medicines"):
    if input_reason:
        predicted_condition, top_5_medicines = recommend_top5_medicines(input_reason)
        st.write(f"Predicted Condition: {predicted_condition}")
        st.write("Top 5 Recommended Medicines:")
        for medicine in top_5_medicines:
            st.write(f"- {medicine}")
    else:
        st.write("Please select a valid medical condition.")

st.header("Find Similar Medicines")
input_medicine = st.text_input("Enter the name of a medicine:")

if st.button("Find Similar Medicines"):
    if input_medicine:
        try:
            similar_medicines = recommend_similar_medicines(input_medicine)
            st.write("Top 5 Similar Medicines:")
            for medicine in similar_medicines:
                st.write(f"- {medicine}")
        except IndexError:
            st.write("Medicine not found. Please enter a valid medicine name.")
    else:
        st.write("Please enter a valid medicine name.")
