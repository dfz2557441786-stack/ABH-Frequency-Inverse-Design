"""Extract fresh positive eigenfrequencies only from the current sample ODB."""
from __future__ import print_function


def extract_positive_frequencies(odb_path, threshold=1.0e-8):
    """Return every strictly positive cyclic eigenfrequency in one fresh ODB."""
    from odbAccess import openOdb
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        frames = odb.steps["Step-1"].frames
        values = []
        for frame in frames:
            frequency = getattr(frame, "frequency", None)
            if frequency is None or frequency <= threshold:
                continue
            values.append(float(frequency))
        return values
    finally:
        odb.close()


def extract_first_positive_frequencies(odb_path, count=6):
    values = extract_positive_frequencies(odb_path)
    if len(values) < count:
        raise RuntimeError("Expected {0} positive frequencies, found {1}.".format(count, len(values)))
    return values[:count]
