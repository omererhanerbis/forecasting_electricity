"""Obtain data from APIs."""
import datetime as dt

import pandas as pd
import requests


def sunlight_data():
    """Request and calculate sunlight time from prayer times API."""
    # if r.status_code == 200:

    is_there_sun = pd.DataFrame(columns=["date", "sunrise", "sunset", "sunlight"])

    for year in range(2017, 2024):
        for month in range(1, 13):
            url = f"http://api.aladhan.com/v1/calendarByCity/{year}/{month}?city=Ankara&country=Turkey&method=13"
            r = requests.get(url)
            if r.status_code == 200:
                resp = r.json()

            list_of_dicts = resp["data"]

            pd.DataFrame(columns=["date", "sunrise", "sunset", "sunlight"])

            date_list = []
            sunrise_list = []
            sunset_list = []
            sunlight_list = []

            for item in list_of_dicts:
                date = item["date"]["gregorian"]["date"]
                date = dt.datetime.strptime(date, "%d-%m-%Y")
                sunrise = item["timings"]["Sunrise"].removesuffix(" (+03)")
                sunset = item["timings"]["Maghrib"].removesuffix(" (+03)")

                sunrise_holder = dt.datetime.strptime(sunrise, "%H:%M")
                sunset_holder = dt.datetime.strptime(sunset, "%H:%M")

                date_list.append(date)
                sunrise_list.append(sunrise)
                sunset_list.append(sunset)

                sunlight = round((sunset_holder - sunrise_holder).seconds / 60)
                sunlight_list.append(sunlight)

            list_of_tuples = list(
                zip(date_list, sunrise_list, sunset_list, sunlight_list),
            )

            list_of_tuples

            is_there_sun_monthly = pd.DataFrame(
                list_of_tuples,
                columns=["date", "sunrise", "sunset", "sunlight"],
            )
            is_there_sun = pd.concat([is_there_sun, is_there_sun_monthly])

    return is_there_sun
