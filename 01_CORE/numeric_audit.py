"""Shared real/finite guards for inputs and derived geometry values."""
from __future__ import division
import math

def assert_real_finite(name, value, context, sample_id='unknown'):
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_real_finite('%s[%d]' % (name, index), item, context, sample_id)
        return value
    if isinstance(value, complex):
        raise ValueError('REAL_FINITE_AUDIT Sample_ID=%s variable_name=%s value=%r real_part=%r imag_part=%r source_expression=%s' % (sample_id, name, value, value.real, value.imag, context))
    try:
        finite = math.isfinite(float(value))
    except (TypeError, ValueError):
        raise ValueError('REAL_FINITE_AUDIT Sample_ID=%s variable_name=%s value=%r real_part=n.a. imag_part=n.a. source_expression=%s' % (sample_id, name, value, context))
    if not finite:
        raise ValueError('REAL_FINITE_AUDIT Sample_ID=%s variable_name=%s value=%r real_part=n.a. imag_part=n.a. source_expression=%s' % (sample_id, name, value, context))
    return value
