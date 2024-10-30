import pandas as pd


def get_match_likelihood(dataframe, key, value):
    yes = len(
        dataframe[(dataframe[key] == value) & (dataframe["Play"] == "Yes")]
    ) / len(dataframe[dataframe["Play"] == "Yes"])
    no = len(dataframe[(dataframe[key] == value) & (dataframe["Play"] == "No")]) / len(
        dataframe[dataframe["Play"] == "No"]
    )
    return yes, no


def get_outlook_match_likelihood(dataframe, value):
    return get_match_likelihood(dataframe, "Outlook", value)


def get_humidity_match_likelihood(dataframe, value):
    return get_match_likelihood(dataframe, "Humidity", value)


def get_wind_match_likelihood(dataframe, value):
    return get_match_likelihood(dataframe, "Wind", value)


def get_probability(*conditions):
    result = 1
    for value in conditions:
        result *= value

    return result


def normalize(yes_probability, no_probability):
    total = yes_probability + no_probability
    yes = yes_probability / total
    no = no_probability / total

    return yes, no


data = pd.DataFrame(
    {
        "Day": [
            "D1",
            "D2",
            "D3",
            "D4",
            "D5",
            "D6",
            "D7",
            "D8",
            "D9",
            "D10",
            "D11",
            "D12",
            "D13",
            "D14",
        ],
        "Outlook": [
            "Sunny",
            "Sunny",
            "Overcast",
            "Rain",
            "Rain",
            "Rain",
            "Overcast",
            "Sunny",
            "Sunny",
            "Rain",
            "Sunny",
            "Overcast",
            "Overcast",
            "Rain",
        ],
        "Humidity": [
            "High",
            "High",
            "High",
            "High",
            "Normal",
            "Normal",
            "Normal",
            "High",
            "Normal",
            "Normal",
            "Normal",
            "High",
            "Normal",
            "High",
        ],
        "Wind": [
            "Weak",
            "Strong",
            "Weak",
            "Weak",
            "Weak",
            "Strong",
            "Strong",
            "Weak",
            "Weak",
            "Weak",
            "Strong",
            "Strong",
            "Weak",
            "Strong",
        ],
        "Play": [
            "No",
            "No",
            "Yes",
            "Yes",
            "Yes",
            "No",
            "Yes",
            "No",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "No",
        ],
    }
)

outlook = "Sunny"
humidity = "High"
wind = "Weak"

yes_outlook_likelihood, no_outlook_likelihood = get_outlook_match_likelihood(
    data,
    outlook,
)
yes_humidity_likelihood, no_humidity_likelihood = get_humidity_match_likelihood(
    data,
    humidity,
)
yes_wind_likelihood, no_wind_likelihood = get_wind_match_likelihood(
    data,
    wind,
)

yes_probability = get_probability(
    yes_outlook_likelihood,
    yes_humidity_likelihood,
    yes_wind_likelihood,
)
no_probability = get_probability(
    no_outlook_likelihood,
    no_humidity_likelihood,
    no_wind_likelihood,
)

yes, no = normalize(yes_probability, no_probability)
print(f"Match WILL happen: {yes:.2f};\nMatch will NOT happen: {no:.2f};")
