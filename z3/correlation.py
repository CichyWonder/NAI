import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr


def find_common_movies(candidate, other_candidate):
    """Do set intersection to find common movies"""
    return set(candidate.recommendations.keys()) & set(other_candidate.recommendations.keys())


def create_common_scoring(candidate, other_candidate, commonMovies):
    """Extracts scoring for common movies and returns it accordingly to the index"""
    candidate_recommendations = []
    other_candidate_recommendations = []

    for movie in commonMovies:
        candidate_recommendations.append(candidate.recommendations[movie])
        other_candidate_recommendations.append(other_candidate.recommendations[movie])

    return np.array(candidate_recommendations, dtype=float), np.array(other_candidate_recommendations, dtype=float)


def create_correlations(chosen_candidate, candidates, algorithm):
    """
    Function responsible for building correlations for both Euclidean Distance and Pearson correlation

    :returns Map where key is name of candidate and value is its correlation with chosen_candidate
    """
    correlations = {}
    for other_candidate in candidates:
        if chosen_candidate != other_candidate:
            common_movies = find_common_movies(chosen_candidate, other_candidate)
            candidate_scoring, other_scoring = create_common_scoring(
                chosen_candidate,
                other_candidate,
                common_movies
            )
            if len(common_movies) > 2:
                dict_entry = {}
                if algorithm == 'pearson':
                    dict_entry = pearson_correlation_entry(other_candidate, candidate_scoring, other_scoring)
                if algorithm == 'euclidean':
                    dict_entry = euclidean_correlation_entry(other_candidate, candidate_scoring, other_scoring)
                correlations.update(dict_entry)

    return correlations


def pearson_correlation_entry(candidate, candidate_scoring, other_scoring):
    correlation, _ = pearsonr(candidate_scoring, other_scoring)
    return {candidate.name: correlation}


def euclidean_correlation_entry(candidate, candidate_scoring, other_candidate_scoring):
    return {candidate.name: euclidean(candidate_scoring, other_candidate_scoring)}
