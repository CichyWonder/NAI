"""
==========================================
Program to recommend movies which user should/ should not watch

Disclaimer:
Some users add shows not movies in database and TMDb api don't have all descriptions

Creators:
Michał Cichowski
Dominik Pasymowski
==========================================
Prerequisites:
Before you run program, you need to install Numpy and TMDb packages.
You can use for example use PIP package manager do to that:
pip install numpy
pip install TMDb
==========================================
Usage:
collaborative_filtering.py --user "Michał Cichowski"

In output you will get:
- list of people who like similar movies as person passed in command
- five recommended movies
- five not recommended movies
==========================================
"""

import argparse
import json
import numpy as np
from tmdbv3api import TMDb, Search

from compute_scores import euclidean_score


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find users who are similar to the input user')
    parser.add_argument('--user', dest='user', required=True,
                        help='Input user')
    return parser


# Finds users in the dataset that are similar to the input user
def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError('Cannot find ' + user + ' in the dataset')

    # Compute Pearson score between input user
    # and all the users in the dataset
    scores = np.array([[x, euclidean_score(dataset, user,
                                           x)] for x in dataset if x != user])

    # Sort the scores in decreasing order
    scores_sorted = np.argsort(scores[:, 1])[::-1]

    # Extract the top 'num_users' scores
    top_users = scores_sorted[:num_users]

    return scores[top_users]


def add_similar_users_to_list(database, similars):
    # saves similar user's names to list
    for item in database:
        for sim_user in similars:
            if item == sim_user[0]:
                similars_users_names.append(item)


def create_films_set():
    # add films from each similar user
    for name in similars_users_names:
        savefilms(data[name])

    # we need to remove lecturer seen movies
    for paul_film in data[user]:
        films.remove(paul_film)



def find_recommended_films():
    # initialize variables for describing recommended films
    score_index = 10

    # going from the top rating, we are looping through all the films
    # we check if users watched shared films and if so, we check their ratings,
    # if it's current searched top rating, we add it to recommended films
    while score_index > 0:
        for film in films:
            for name in similars_users_names:
                if len(recommended_films) < 5:
                    if film in data[name] and data[name][film] == score_index:
                        recommended_films.add(film)
        score_index -= 1


def find_not_recommended_films():
    # initialize variables for describing recommended films
    score_index = 1

    # going from the top rating, we are looping through all the films
    # we check if users watched shared films and if so, we check their ratings,
    # if it's current searched top rating, we add it to recommended films
    while score_index < 11:
        for film in films:
            for name in similars_users_names:
                if len(not_recommended_films) < 5:
                    if film in data[name] and data[name][film] == score_index:
                        not_recommended_films.add(film)
        score_index += 1


# function to save films to set
def savefilms(user_data):
    for film in user_data:
        films.add(film)


# function to print results with plot from imdb(film's library) package
def print_title_and_plot(film_set_to_print):

    for film in film_set_to_print:
        movie_description = search.movies({"query": film})
        print("'",film,"'")  # we print plot description

        for result in movie_description:
            print(result.overview) # we print plot description

def calculate_user_distance():
    i = len(similar_users) - 1
    while i > 0:
        x = float(similar_users[0][1])
        y = float(similar_users[i][1])
        if x - acceptable_diff > y:
            user_dist.append(similar_users[0])
        else:
            user_dist.append(similar_users[i])
        i -= 1


if __name__ == '__main__':

    tmdb = TMDb()  # creates instance of TMDb
    tmdb.api_key = '9b50be5e6a7ba1bac1ec45666cd02d20'
    tmdb.language = 'pl'
    search = Search()

    recommended_films = set() # we dont want repeated films
    not_recommended_films = set()  # we dont want repeated films
    acceptable_diff = 0.1  # acceptable diff

    similars_users_names = []
    user_dist = []
    films = set()  # we dont want repeated films
    args = build_arg_parser().parse_args()
    user = args.user

    ratings_file = 'data.json'

    with open(ratings_file, 'r', encoding="utf8") as f:
        data = json.loads(f.read())

    savefilms(data[user])

    print('\nWynik podobieństwa dla użytkownika: ' + user + '\n')
    similar_users = find_similar_users(data, user, 3)

    print('Jesteś podobny do użytkowników')
    print('-' * 41)

    for item in similar_users:
        print(item[0], '\t\t', round(float(item[1]), 2))

    calculate_user_distance()

    add_similar_users_to_list(data, similar_users)
    create_films_set()
    find_recommended_films()
    find_not_recommended_films()

    print('\nRekomendowane Filmy: ')
    print('-' * 41)
    print_title_and_plot(recommended_films)
    print('-' * 41)
    print('\nNie Rekomendowane filmy: ')
    print('-' * 41)
    print_title_and_plot(not_recommended_films)
    print('-' * 41)
