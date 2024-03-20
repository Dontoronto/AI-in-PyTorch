
from operator import itemgetter

def fixed_reduction_rate(available_pattern_indices, pattern_distribution, min_amount_indices):
    '''
    This method reduces the available/ choosable of the patttern library.
    Target is to minimize the amount of pattern selection for all conv layers to have larger groups of
    similar patterns. For further regrouping and performance enhancing on embedded devices.
    With hardware optimization
    :param available_pattern_indices: list with values which correspond with the indexes of the pattern
            library
    :param pattern_distribution: list which correspond with the pattern library. Has the aggregated amount
            of every used pattern at the same indexposition as the pattern library
    :param min_amount_indices: int value. threshold value until which amount the available_pattern_indices
            list is reducable. Papers suggest the values 12,8 or 6
    :return: returns the index of the pattern with the least occurences in between pattern library
            if values are duplicate it picks the first value occurence
    '''
    if len(available_pattern_indices) > min_amount_indices:
        sliced_list = itemgetter(*available_pattern_indices)(pattern_distribution)

        lowest_index = index_of_lowest_value(sliced_list)

        available_pattern_indices.pop(lowest_index)

        return lowest_index

def index_of_lowest_value(lst):
    '''
    this function gives back the index of the smallest value in list.
    :param lst: list of numbers
    :return: index of the smallest value
    '''
    return lst.index(min(lst)) if lst else None