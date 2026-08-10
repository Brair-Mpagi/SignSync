from __future__ import annotations

import pytest

from signsync.vision.synthetic import SignerStyle, synthetic_sign

#: A small vocabulary used across the test suite. These are placeholders for
#: pipeline testing, not USL entries — see docs/limitations.md.
DEMO_GLOSSES = ["HELLO", "HELP", "HOSPITAL", "WATER", "THANK-YOU", "NAME"]


@pytest.fixture
def signer() -> SignerStyle:
    return SignerStyle.derived("signer-a")


@pytest.fixture
def other_signer() -> SignerStyle:
    return SignerStyle.derived("signer-b")


@pytest.fixture
def clip(signer: SignerStyle):
    return synthetic_sign("HELLO", signer)
