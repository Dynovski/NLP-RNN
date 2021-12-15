class IncorrectNumberOfAttributesException(Exception):
    def __init__(self, expected: int, actual: int):
        super(
            IncorrectNumberOfAttributesException,
            self
        ).__init__(f'Expected list of {expected} attributes, got: {actual}')
