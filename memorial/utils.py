"""For fancy printing."""

from __future__ import annotations

from typing import Any, Literal

# colour list
c_colors = {
    "HEADER": "\033[95m",
    "OKBLUE": "\033[94m",
    "OKCYAN": "\033[96m",
    "OKGREEN": "\033[92m",
    "WARNING": "\033[93m",
    "FAIL": "\033[91m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
}
end_c = "\033[0m"


def cstr(
    x: Any,
    ctype: Literal[
        "HEADER", "OKBLUE", "OKCYAN", "OKGREEN", "WARNING", "FAIL", "BOLD", "UNDERLINE"
    ],
) -> str:
    """Makes a string colourful.

    Args:
        x (Any): the string
        ctype (str): the colour

    Returns:
        str: the coloured string

    """
    return f"{c_colors[ctype]}{x}{end_c}"


class MemorialException(Exception):
    """ReplayBufferException."""

    def __init__(self, message: str = ""):
        """__init__.

        Args:
            message (str): the message

        """
        message = cstr(message, "FAIL")
        super().__init__(message)
        self.message = message
