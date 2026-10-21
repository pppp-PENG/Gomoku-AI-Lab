"""Intentional permission gate, not a claimed or unverified XML game parser.

The official page offers XML/XML.GZ; its restriction to noncommercial OFFLINE
databases does not authorise this shared web-based SFT workflow. No bulk file was
requested, so XML element semantics, coordinate orientation and rules are unverified.
"""


def records(path, provenance):
    raise PermissionError("RenjuNet: noncommercial offline database use only; shared training corpus blocked")
