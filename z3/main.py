from engine import find_recommended_movies_for_candidate, find_not_recommended_movies_for_candidate

candidate_name = input("Enter your name: ")
algorithm = input("Enter algorithm [Pearson, Euclidean]:")

if algorithm.lower() != 'pearson' and algorithm.lower() != 'euclidean':
    raise ValueError(f'Unexpected value: {algorithm}')

Precommended_movies = find_recommended_movies_for_candidate(candidate_name, algorithm.lower())
pearson_not_recommended_movies = find_not_recommended_movies_for_candidate(candidate_name, algorithm.lower())

print("Recommended:")
print(recommended_movies)
print("Not recommended:")
print(pearson_not_recommended_movies)
