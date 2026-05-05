from enum import IntEnum
from typing import List


class Vocabulary(IntEnum):
    """
    Marker base for CA vocabularies. Empty so concrete vocabularies can subclass and
    declare their own members (Python forbids extending an Enum once members exist).
    """
    pass


class NeuralCellularAutomatonVocabulary(Vocabulary):
    IGNORE = 0
    EMPTY = 1

    TRUE = 2
    FALSE = 3

    INTERMEDIATE_TRUTH_VALUE = 4

    INPUT_NEURON = 5

    HIDDEN_NEURON_1 = 6
    HIDDEN_NEURON_2 = 7
    HIDDEN_NEURON_3 = 8
    HIDDEN_NEURON_4 = 9

    SYNAPSE = 10

    OUTPUT_NEURON = 11

    VERTICAL_BOUNDARY = 12


def hidden_neuron_token_values(vocabulary_cls, num_types: int) -> List[int]:
    """
    Int token values for the first ``num_types`` hidden-neuron classes (1 → only
    `HIDDEN_NEURON_1`, 4 → `HIDDEN_NEURON_1`..`4`). Use for rewards, logit priors, and
    layout masking.
    """
    if num_types not in (1, 4):
        raise ValueError(f"num_hidden_neuron_types must be 1 or 4, got {num_types}")
    hs = (
        vocabulary_cls.HIDDEN_NEURON_1,
        vocabulary_cls.HIDDEN_NEURON_2,
        vocabulary_cls.HIDDEN_NEURON_3,
        vocabulary_cls.HIDDEN_NEURON_4,
    )
    return [int(t.value) for t in hs[:num_types]]


# Colors for visualization (not used in the automaton logic)
VOCABULARY_COLORS = {
    NeuralCellularAutomatonVocabulary.EMPTY.value: (0, 0, 0),
    NeuralCellularAutomatonVocabulary.TRUE.value: (0, 0, 255),
    NeuralCellularAutomatonVocabulary.FALSE.value: (255, 0, 0),
    NeuralCellularAutomatonVocabulary.INTERMEDIATE_TRUTH_VALUE.value: (255, 255, 0),
    NeuralCellularAutomatonVocabulary.INPUT_NEURON.value: (0, 255, 0),
    NeuralCellularAutomatonVocabulary.HIDDEN_NEURON_1.value: (255, 0, 255),
    NeuralCellularAutomatonVocabulary.HIDDEN_NEURON_2.value: (128, 128, 255),
    NeuralCellularAutomatonVocabulary.HIDDEN_NEURON_3.value: (255, 128, 128),
    NeuralCellularAutomatonVocabulary.HIDDEN_NEURON_4.value: (255, 255, 128),
    NeuralCellularAutomatonVocabulary.SYNAPSE.value: (255, 255, 255),
    NeuralCellularAutomatonVocabulary.OUTPUT_NEURON.value: (0, 255, 255),
    NeuralCellularAutomatonVocabulary.VERTICAL_BOUNDARY.value: (128, 128, 128),
}
