import json
import traceback

import joblib
import pandas as pd
from numpy import dot
from numpy.linalg import norm

# read the data
df = pd.read_csv("products.csv")

# Keep only the important columns
df1 = df[["id", "unit_price", "chest", "waist", "arm_inseam", "leg_inseam", "arm_circumference", "leg_circumference",
          "material", "color", "season", "style", "sex", "is_kid_friendly"]]

# Create dummies for categorical columns
df_material_dummies = df1["material"].str.get_dummies(sep=",")
df_color_dummies = df1["color"].str.get_dummies(sep=",")
df_season_dummies = df1["season"].str.get_dummies(sep=",")
df_style_dummies = df1["style"].str.get_dummies(sep=",")
df_sex_dummies = df1["sex"].str.get_dummies()

merged_dummies = pd.concat([df_material_dummies, df_color_dummies, df_season_dummies, df_style_dummies, df_sex_dummies],
                           axis=1)
df_result = pd.concat([df1, merged_dummies], axis=1)

# Drop the old categorical columns because dummy columns are created
df_final = df_result.drop(['color', "material", "season", "style", "sex"], axis=1)

# Transform the dataframe to a numpy array to feed it to our recommendations function
matrix = df_final.to_numpy()

# Creating the Recommendations function
i = 0
# list = []
dict = {}


def get_recoms(user_ideal_product):
    for item in matrix:
        # list.append(item[1:28])
        if item[27] != user_ideal_product[26]:
            # list.append(0)
            dict[item[0]] = 0
        else:
            # list.append(dot(user_ideal_product, item)/(norm(user_ideal_product)*norm(item)))
            dict[item[0]] = dot(user_ideal_product, item[1:28]) / (norm(user_ideal_product) * norm(item[1:28]))

    # return list
    return dict


# Save my Recommendations function
joblib.dump(get_recoms, 'model.pkl')
print("Model dumped!")

model = joblib.load('model.pkl')

# Save the columns of df_final
model_columns = list(df_final.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")

# print(get_recoms([300, 93.7, 81, 64, 29, 28.75, 55, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]))

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/predict", methods=['POST', "GET"])
def predict():
    if model:
        try:
            json_ = request.json
            print(model_columns)
            print(json_)
            print(pd.json_normalize(json_))

            query_data = pd.json_normalize(json_)


            # Process query data
            query_material_dummies = query_data["material"].str.get_dummies(sep=",")
            query_color_dummies = query_data["color"].str.get_dummies(sep=",")
            query_season_dummies = query_data["season"].str.get_dummies(sep=",")
            query_style_dummies = query_data["style"].str.get_dummies(sep=",")
            query_sex_dummies = query_data["sex"].str.get_dummies()

            query_merged_dummies = pd.concat(
                [query_material_dummies, query_color_dummies, query_season_dummies, query_style_dummies,
                 query_sex_dummies], axis=1)
            query_data_result = pd.concat([query_data, query_merged_dummies], axis=1)

            query_data_final = query_data_result.drop(['color', "material", "season", "style", "sex"], axis=1)

            print(query_data_final)


            df_precious=query_data_final.reindex(model_columns,axis='columns',fill_value=0)



            row = df_precious.to_numpy()
            print(row[0])

            json_object=json.dumps(model(row[0][1:28]))

            return json_object

        except:
            # print("HEYY",json_)
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return 'No model here to use'


if __name__ == '__main__':
    model = joblib.load("model.pkl")  # Load "model.pkl"
    print('Model loaded')

    model_columns = joblib.load("model_columns.pkl")  # Load "model_columns.pkl"
    print('Model columns loaded')

    app.run(debug=True)
