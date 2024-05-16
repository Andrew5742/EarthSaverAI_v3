# -*- coding: utf-8 -*-

import csv

import matplotlib
from django.conf import settings
import os
from django.shortcuts import render, redirect
from django.http import HttpResponse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO
import base64
from keras.models import load_model
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
#from tensorflow.keras.models import load_model
from keras.models import load_model
import keras
import tensorflow as tf

matplotlib.use('Agg')


# Create your views here.
def index(request):
    years = range(2024, 2101)

    df = read_data()
    country_names = df['Country'].unique().tolist()
    if request.method == 'POST':
        # Handle form submission
        # Redirect to plot_predicted_data view
        return redirect('plot_predicted_data', {'years': years, 'country_names': country_names})
    else:
        return render(request, 'main/index.html', {'years': years, 'country_names': country_names})


def plot_predicted_data(request):
    if request.method == 'POST':
        country = request.POST.get('country', 'Australia')
        year = request.POST.get('year', '2024')
    else:
        # If the view is accessed directly without POST data, set default country
        country = 'Australia'
        year = '2024'

    # Read data from CSV
    df = read_data()
    new_data = df[df['Country'] == country]

    last_dt = new_data['dt'].iloc[-1]
    last_dt = pd.to_datetime(last_dt)
    month_number = last_dt.month

    climate_zone = new_data['MainClimateZone'].iloc[0]

    if climate_zone == 'A':
        model = load_model('data/model_A.h5')
    elif climate_zone == 'B':
        model = load_model('data/model_B.h5')
    elif climate_zone == 'C':
        model = load_model('data/model_C.h5')
    elif climate_zone == 'D':
        model = load_model('data/model_D.h5')
    else:
        model = load_model('data/model_E.h5')

    path3 = os.path.join(settings.PROJECT_ROOT, 'data', 'scaler.pkl')

    with open(path3, 'rb') as f:
        scaler = pickle.load(f)

    new_data['dt'] = pd.to_datetime(new_data['dt'])
    new_data = new_data.set_index('dt')

    new_data['months'] = (new_data.index.year - 1970) * 12 + new_data.index.month

    new_data.dropna(subset=['AverageTemperature'], inplace=True)

    scaled_new_data = scaler.transform(new_data['AverageTemperature'].values.reshape(-1, 1))

    x_new = []
    for i in range(30, len(scaled_new_data)):
        x_new.append(scaled_new_data[i - 30:i, 0])
    x_new = np.array(x_new)
    x_new = np.reshape(x_new, (x_new.shape[0], x_new.shape[1], 1))

    new_prediction = model.predict(x_new)

    new_prediction = scaler.inverse_transform(new_prediction)

    # print(new_prediction)
    predicted_temperature = new_prediction
    predicted_temperature = predicted_temperature[12 - month_number:]

    n = len(predicted_temperature)
    remainder = n % 12

    if remainder != 0:
        predicted_temperature = predicted_temperature[:-remainder]

    predicted_temperature = predicted_temperature.reshape(-1, 12)

    # print(predicted_temperature)
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    month_range = range(12)
    table_data = predicted_temperature[int(year) - 2014][:12]
    month_and_data = zip(months, table_data)

    months_abbreviations = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    df_2000 = df[(df['Country'] == country) & (df['dt'].dt.year == 2000)]
    avg_temp_2000 = df_2000['AverageTemperature'].values

    df_1900 = df[(df['Country'] == country) & (df['dt'].dt.year == 1900)]
    avg_temp_1900 = df_1900['AverageTemperature'].values

    plt.figure(figsize=(8, 4))
    plt.plot(predicted_temperature[int(year) - 2014], color='red', label=f'Predicted Temperature in {year}')
    plt.plot(avg_temp_2000, color='blue', label='Average Temperature in 2000')
    plt.plot(avg_temp_1900, color='green', label='Average Temperature in 1900')
    plt.title(f'Temperature Prediction for {year} in {country}')
    plt.xlabel('Month')
    plt.ylabel('Temperature')
    plt.xticks(range(12), months_abbreviations)
    plt.legend()
    plt.grid(True)

    # Convert plot to base64 encoded image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    average_temp = np.mean(table_data)
    df_2000 = df[(df['Country'] == country) & (df['dt'].dt.year == 2000)]
    average_temp_2000 = df_2000['AverageTemperature'].mean()
    df_1900 = df[(df['Country'] == country) & (df['dt'].dt.year == 1900)]
    average_temp_1900 = df_1900['AverageTemperature'].mean()

    dif_2000 = average_temp - average_temp_2000
    dif_1900 = average_temp - average_temp_1900

    # Prepare context for template
    context = {
        'country': country,
        'year': year,
        'dif_2000': dif_2000,
        'dif_1900': dif_1900,
        'month_and_data': month_and_data,
        'plot_data_uri': plot_data_uri
    }

    # Convert plot to base64 string
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Embed plot data into an HTML img tag
    html_plot = f'<img src="data:image/png;base64,{plot_data}" />'

    # return HttpResponse(html_plot)
    return render(request, 'main/plot_template.html', context)


def read_data():
    csv_path1 = os.path.join(settings.PROJECT_ROOT, 'data', 'GlobalLandTemperaturesByCountry.csv')

    df = pd.read_csv(csv_path1)

    # Convert 'dt' to datetime format
    df['dt'] = pd.to_datetime(df['dt'])

    replacement_dict = {
        "Antigua And Barbuda": "Antigua and Barbuda",
        "Bahamas": "The Bahamas",
        "Bosnia And Herzegovina": "Bosnia and Herzegovina",
        "Bonaire, Saint Eustatius And Saba": "Bonaire",
        "Burma": "Myanmar",
        "Côte D'Ivoire": "Cote D'Ivoire",
        "Curaçao": "Curacao",
        "Macedonia": "North Macedonia",
        "Congo (Democratic Republic Of The)": "Democratic Republic of the Congo",
        "Congo": "Republic of the Congo",
        "Gambia": "The Gambia",
        "Guinea Bissau": "Guinea-Bissau",
        "Federated States Of Micronesia": "Federated States of Micronesia",
        "French Southern And Antarctic Lands": "French Southern and Antarctic Lands",
        "Heard Island And Mcdonald Islands": "Heard Island and Mcdonald Islands",
        "Isle Of Man": "Isle of Man",
        "Palestina": "Palestine",
        "Palmyra Atoll": "United States Minor Outlying Islands",
        "Saint Barthélemy": "Saint Barthelemy",
        "Saint Kitts And Nevis": "Saint Kitts and Nevis",
        "Saint Martin": "North Macedonia",
        "Saint Pierre And Miquelon": "Saint Pierre and Miquelon",
        "Saint Vincent And The Grenadines": "Saint Vincent and the Grenadines",
        "Sao Tome And Principe": "Sao Tome and Principe",
        "Sint Maarten": "Saint Martin",
        "South Georgia And The South Sandwich Isla": "South Georgia and South Sandwich Islands",
        "Swaziland": "Eswatini",
        "Timor Leste": "Timor-Leste",
        "Trinidad And Tobago": "Trinidad and Tobago",
        "Turks And Caicas Islands": "Turks and Caicos Islands",
        "Virgin Islands": "US Virgin Islands"
    }

    df['Country'] = df['Country'].replace(replacement_dict)

    climate_zones = {
        'Åland': 'DFB',
        'Afghanistan': 'DSB',
        'Albania': 'CSB',
        'Algeria': 'BWH',
        'American Samoa': 'AW',
        'Andorra': 'CFB',
        'Angola': 'AW',
        'Anguilla': 'AW',
        'Antigua and Barbuda': 'AF',
        'Argentina': 'CFA',
        'Armenia': 'DFB',
        'Aruba': 'AW',
        'Australia': 'BWH',
        'Austria': 'DFB',
        'Azerbaijan': 'BSK',
        'Bahrain': 'BWH',
        'Bangladesh': 'AW',
        'Baker Island': 'AF',
        'Barbados': 'AM',
        'Belarus': 'DFB',
        'Belgium': 'CFB',
        'Belize': 'AM',
        'Benin': 'AW',
        'Bermuda': 'CFA',
        'Bhutan': 'CWB',
        'Bolivia': 'AW',
        'Bonaire': 'BS',
        'Bosnia and Herzegovina': 'DFB',
        'Botswana': 'BSH',
        'Brazil': 'AW',
        'British Indian Ocean Territory': 'AM',
        'British Virgin Islands': 'AM',
        'Brunei': 'AF',
        'Bulgaria': 'DFB',
        'Burkina Faso': 'BSH',
        'Burundi': 'CWB',
        'Cambodia': 'AM',
        'Cameroon': 'AW',
        'Canada': 'DFC',
        'Cape Verde': 'BWH',
        'Cayman Islands': 'AW',
        'Central African Republic': 'AW',
        'Chad': 'CFA',
        'Chile': 'CSB',
        'China': 'BSK',
        'Christmas Island': 'AW',
        'Cocos (Keeling) Islands': 'AF',
        'Colombia': 'CFB',
        'Comoros': 'AF',
        'Costa Rica': 'CFB',
        "Cote D'Ivoire": 'AW',
        'Croatia': 'CFB',
        'Cuba': 'AW',
        'Curacao': 'AW',
        'Cyprus': 'BSH',
        'Curaçao': 'AW',
        'Czech Republic': 'CFB',
        'Democratic Republic of the Congo': 'AM',
        'Denmark': 'CFB',
        'Denmark (Europe)': 'CFB',
        'Djibouti': 'BWH',
        'Dominica': 'AF',
        'Dominican Republic': 'AF',
        'Ecuador': 'CFB',
        'Egypt': 'BWH',
        'El Salvador': 'AW',
        'Equatorial Guinea': 'AM',
        'Eritrea': 'BWH',
        'Estonia': 'DFB',
        'Eswatini': 'CWA',
        'Ethiopia': 'BSH',
        'Falkland Islands (Islas Malvinas)': 'ET',
        'Faroe Islands': 'CFC',
        'Federated States of Micronesia': 'AF',
        'Fiji': 'AF',
        'Finland': 'DFC',
        'France': 'CFB',
        'France (Europe)': 'CFB',
        'French Guiana': 'AF',
        'French Polynesia': 'AF',
        'French Southern and Antarctic Lands': 'CFC',
        'French Southern Territories': 'ET',
        'Gaza Strip': 'BSH',
        'Gabon': 'AW',
        'Georgia': 'CFA',
        'Germany': 'CFB',
        'Ghana': 'AW',
        'Gibraltar': 'CSA',
        'Greece': 'CSA',
        'Greenland': 'EF',
        'Grenada': 'AW',
        'Guadeloupe': 'AF',
        'Guam': 'AF',
        'Guatemala': 'AM',
        'Guernsey': 'CFB',
        'Guinea': 'AW',
        'Guinea-Bissau': 'AW',
        'Guyana': 'AF',
        'Haiti': 'AW',
        'Heard Island and Mcdonald Islands': 'ET',
        'Honduras': 'AM',
        'Hong Kong': 'CWA',
        'Hungary': 'CFB',
        'Iceland': 'ET',
        'India': 'AW',
        'Indonesia': 'AF',
        'Iran': 'BWK',
        'Iraq': 'BWH',
        'Ireland': 'CFB',
        'Isle of Man': 'CFB',
        'Israel': 'BSH',
        'Italy': 'CSA',
        'Jamaica': 'AW',
        'Japan': 'DFB',
        'Jersey': 'CFB',
        'Jordan': 'CFA',
        'Kazakhstan': 'BSK',
        'Kingman Reef': 'AF',
        'Kenya': 'AW',
        'Kiribati': 'AF',
        'Kosovo': 'CFB',
        'Kuwait': 'BWH',
        'Kyrgyzstan': 'DFB',
        'Laos': 'CWA',
        'Latvia': 'DFB',
        'Lebanon': 'CSA',
        'Lesotho': 'CWB',
        'Liberia': 'AM',
        'Libya': 'BWH',
        'Liechtenstein': 'CFB',
        'Lithuania': 'DFB',
        'Luxembourg': 'CFB',
        'Macau': 'CWA',
        'Madagascar': 'CWB',
        'Malawi': 'AW',
        'Malaysia': 'AF',
        'Maldives': 'AM',
        'Mali': 'BWH',
        'Malta': 'CSA',
        'Marshall Islands': 'AF',
        'Martinique': 'AF',
        'Mauritania': 'BWH',
        'Mauritius': 'AM',
        'Mayotte': 'AW',
        'Mexico': 'BSK',
        'Moldova': 'DFB',
        'Monaco': 'CSB',
        'Mongolia': 'BSK',
        'Montenegro': 'CFB',
        'Montserrat': 'AM',
        'Morocco': 'BSH',
        'Mozambique': 'AW',
        'Myanmar': 'AW',
        'Namibia': 'BWH',
        'Nauru': 'AF',
        'Nepal': 'CWB',
        'Netherlands': 'CFB',
        'Netherlands (Europe)': 'CFB',
        'New Caledonia': 'CFA',
        'New Zealand': 'CFB',
        'Nicaragua': 'AM',
        'Niger': 'BWH',
        'Nigeria': 'AW',
        'Niue': 'AF',
        'Northern Mariana Islands': 'AF',
        'North Korea': 'DWB',
        'North Macedonia': 'CFB',
        'Norway': 'DFC',
        'Oman': 'BWH',
        'Pakistan': 'BWH',
        'Palau': 'AF',
        'Palestine': 'CSA',
        'Panama': 'AM',
        'Papua New Guinea': 'AF',
        'Paraguay': 'AW',
        'Peru': 'AF',
        'Philippines': 'AF',
        'Poland': 'CFB',
        'Portugal': 'CSA',
        'Puerto Rico': 'AM',
        'Qatar': 'BWH',
        'Republic of the Congo': 'AM',
        'Reunion': 'AS',
        'Romania': 'DFB',
        'Russia': 'DFC',
        'Rwanda': 'CFB',
        'Saint Barthelemy': 'AW',
        'Saint Kitts and Nevis': 'AW',
        'Saint Lucia': 'AF',
        'Saint Martin': 'AW',
        "Saint Pierre and Miquelon": 'DFB',
        'Saint Vincent and the Grenadines': 'AF',
        'Samoa': 'AW',
        'San Marino': 'CFB',
        'Sao Tome and Principe': 'AS',
        'Saudi Arabia': 'BWH',
        'Senegal': 'BSH',
        'Serbia': 'CFB',
        'Seychelles': 'AF',
        'Sierra Leone': 'AM',
        'Singapore': 'AF',
        'Slovakia': 'DFB',
        'Slovenia': 'CFB',
        'Solomon Islands': 'AF',
        'Somalia': 'BWH',
        'South Africa': 'BWK',
        'South Georgia and South Sandwich Islands': 'ET',
        'South Korea': 'DWB',
        'South Sudan': 'AW',
        'Spain': 'CSA',
        'Sri Lanka': 'AF',
        'Sudan': 'BSH',
        'Suriname': 'AF',
        'Svalbard And Jan Mayen': 'ET',
        'Sweden': 'DFB',
        'Switzerland': 'DFC',
        'Syria': 'BWH',
        'Taiwan': 'CWB',
        'Tajikistan': 'DSC',
        'Tanzania': 'BSH',
        'Thailand': 'AW',
        'The Bahamas': 'AW',
        'The Gambia': 'AW',
        'Timor-Leste': 'AW',
        'Togo': 'CFA',
        'Tonga': 'AF',
        'Trinidad and Tobago': 'AM',
        'Tunisia': 'BWH',
        'Turkey': 'CSB',
        'Turkmenistan': 'BSK',
        'Turks and Caicos Islands': 'AS',
        'Tuvalu': 'AF',
        'Uganda': 'AM',
        'Ukraine': 'DFB',
        'United Arab Emirates': 'BWH',
        'United Kingdom': 'CFB',
        'United Kingdom (Europe)': 'CFB',
        'United States': 'CFA',
        'United States Minor Outlying Islands': 'AF',
        'Uruguay': 'CFA',
        'US Virgin Islands': 'AM',
        'Uzbekistan': 'BWK',
        'Vanuatu': 'AF',
        'Vatican': 'CSA',
        'Venezuela': 'AM',
        'Vietnam': 'CWA',
        'Western Sahara': 'BWH',
        'Yemen': 'BWH',
        'Zambia': 'CWA',
        'Zimbabwe': 'BSH'
    }

    # Use the map function to create a new column 'ClimateZone'
    df['ClimateZone'] = df['Country'].map(climate_zones)
    df['MainClimateZone'] = df['ClimateZone'].str[0]
    df = df.dropna(subset=['MainClimateZone'])
    return df
