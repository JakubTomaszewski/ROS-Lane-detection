class IncorrectImage(Exception):
    """Class representing an incorrect image exception"""

    def __init__(self, message=''):
        super().__init__(message)
        self.message = message
