class QueryError(Exception):
    """Base class for query-related errors."""


class EmptyQueryError(QueryError):
    pass


class MissingColumnsError(QueryError):
    def __init__(self, missing_cols: list) -> None:
        """
        Initialize the MissingColumnsError.

        Parameters
        ----------
        missing_cols : list
            List of missing column names.
        """
        msg = f"The following columns are missing from the DataFrame: {missing_cols}"
        super().__init__(msg)
        self.missing_cols = missing_cols


class UserFacingError(Exception):
    """
    An error whose message tells the user what is wrong and what to change.

    The GUI shows it in a message box, without a traceback (see
    :func:`celldetective.gui.base.components.show_user_errors`).
    """


class MovieChannelsMismatchError(UserFacingError, ValueError):
    """The experiment declares more channels than a movie has per frame."""
