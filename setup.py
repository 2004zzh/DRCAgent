"""Mark the release distribution as platform-specific.

The core extensions are shipped prebuilt rather than compiled by setuptools.
Without this marker, wheel incorrectly labels the archive as ``py3-none-any``.
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return True


setup(distclass=BinaryDistribution)
