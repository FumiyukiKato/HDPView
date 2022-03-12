class Query:
    """Range Query representation

        Query have some QueryConditions
    """
    def __init__(self, conditions):
        """
        Args:
            conditions (QueryCondition[]): list of QueryCondition
        """
        self.conditions = conditions
        
    def show(self):
        """Print query information.

        Examples:
            >>> Query([ QueryCondition(0, 2, 3), QueryCondition(1, 0, 0) ]).show()
            [(0, 2, 3), (1, 0, 0)]
        
        """
        print([(condition.attribute, condition.start, condition.end) for condition in self.conditions])


class QueryCondition:
    """Range Query condition

        Attributes:
            attribute (int): dimension or attribute
            start (int): start of range at self.attribute
            end (int): end of range at self.attribute
    """    
    def __init__(self, attribute, start, end):
        self.attribute = attribute
        self.start = start
        self.end = end