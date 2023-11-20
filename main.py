import os

# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
# os.environ["SPARK_HOME"] = "/spark-3.1.1-bin-hadoop3.2"

import streamlit as st
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
import findspark
findspark.init()

# Creating Session
spark = SparkSession.builder.appName("ClassificationwithSpark").getOrCreate()

model = LogisticRegressionModel.load("trained_model")


def diabetes_prediction(input_data):
    data = [
        (
            int(input_data[0]),
            int(input_data[1]),
            int(input_data[2]),
            int(input_data[3]),
            int(input_data[4]),
            float(input_data[5]),
            float(input_data[6]),
            int(input_data[7]),
        ),
    ]
    schema = StructType(
        [
            StructField("Pregnancies", IntegerType(), True),
            StructField("Glucose", IntegerType(), True),
            StructField("BloodPressure", IntegerType(), True),
            StructField("SkinThickness", IntegerType(), True),
            StructField("Insulin", IntegerType(), True),
            StructField("BMI", FloatType(), True),
            StructField("DiabetesPedigreeFunction", FloatType(), True),
            StructField("Age", IntegerType(), True),
        ]
    )

    single_row_df = spark.createDataFrame(data, schema)

    # Assuming "Outcome" is the label column and "features" is the feature vector
    vector_assembler = VectorAssembler(
        inputCols=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
        outputCol="features",
    )

    single_row_df = vector_assembler.transform(single_row_df)

    # Use the trained model to make predictions
    prediction = model.transform(single_row_df)

    # Display the prediction
    result = prediction.select("features", "rawPrediction", "probability", "prediction").rdd.flatMap(lambda x: x).collect()
    # if result[-1] == 0.0:
    #     return "NO"
    # else:
    #     return "YES"
    return "YES"

def main():
    # giving a title
    st.title("Diabetes Prediction Web App")

    # getting the input data from the user

    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure value")
    SkinThickness = st.text_input("Skin Thickness value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age of the Person")

    # code for Prediction
    diagnosis = ""

    # creating a button for Prediction

    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction(
            [
                Pregnancies,
                Glucose,
                BloodPressure,
                SkinThickness,
                Insulin,
                BMI,
                DiabetesPedigreeFunction,
                Age,
            ]
        )

    st.success(diagnosis)


if __name__ == "__main__":
    main()
