class QueryError(Exception):
    """Base class for query-related errors."""

class EmptyQueryError(QueryError):
    pass

class MissingColumnsError(QueryError):
    def __init__(self, missing_cols):
        msg = f"The following columns are missing from the DataFrame: {missing_cols}"
        super().__init__(msg)
        self.missing_cols = missing_cols
