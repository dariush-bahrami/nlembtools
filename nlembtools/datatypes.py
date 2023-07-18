from hashlib import sha512
from typing import TypedDict


class TripletDict(TypedDict):
    sha512_hash: str
    anchor: str
    positive: str
    negative: str


class Triplet:
    """This class represents a triplet of strings, consisting of an anchor, a positive
    and a negative string. The anchor and positive strings are similar, while the
    negative string is dissimilar to the anchor string. The reason for creating this
    class is to handle triplets with the same anchor and positive strings but different
    negative strings as equal. This is useful for creating a set of triplets. The
    `to_dict` method can be used to convert a Triplet object to a dictionary, which can
    be used to store the triplet in a database. The `sha512_hash` key of the dictionary
    is a hash of the triplet, which can be used to check if two triplets are equal.

    Args:
        anchor (str): The anchor string.
        positive (str): The positive string.
        negative (str): The negative string.

    Examples:
        >>> triplet1 = Triplet("hello", "hi", "goodbye")
        >>> triplet2 = Triplet("hi", "hello", "goodbye")
        >>> print(triplet1 == triplet2)
    """

    def __init__(self, anchor, positive, negative):
        self.__anchor = anchor
        self.__positive = positive
        self.__negative = negative

    @property
    def anchor(self):
        return self.__anchor

    @property
    def positive(self):
        return self.__positive

    @property
    def negative(self):
        return self.__negative

    def to_dict(self):
        return {
            "sha512_hash": sha512(str(hash(self)).encode("utf-8")).hexdigest(),
            "anchor": self.anchor,
            "positive": self.positive,
            "negative": self.negative,
        }

    def __repr__(self):
        return f"Triplet({self.anchor}, {self.positive}, {self.negative})"

    def __hash__(self):
        return hash((frozenset({self.anchor, self.positive}), self.negative))

    def __eq__(self, other):
        if not isinstance(other, Triplet):
            return False
        else:
            return hash(self) == hash(other)
