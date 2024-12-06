import json
from dataclasses import dataclass
from typing import Dict


@dataclass
class Candidate:
    """Class for keeping track of a candidate with his movie recommendations"""
    name: str
    recommendations: Dict[str, float]

    @staticmethod
    def from_json_data(candidate):
        name = candidate['name']
        recommendations = {}
        for recommendation in candidate['recommendations']:
            recommendations.update({recommendation['movie']: recommendation['score']})

        return Candidate(name, recommendations)


def initialize_data_from_json():
    """Open and load json data from data.json"""
    with open('data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        print(data)

    """Initialize array of candidates objects by mapping each json record to python object"""
    candidates = []
    for jsonCandidate in data['people']:
        candidate = Candidate.from_json_data(jsonCandidate)
        candidates.append(candidate)

    return candidates
