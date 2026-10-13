"""Strict, opt-in cleanup for completed Abaqus job directories.

This module intentionally accepts explicit directories only.  It never scans a
drive, follows links, or deletes an ambiguous folder.  Failed jobs are always
preserved because their diagnostic files remain useful.
"""
from __future__ import print_function

import os


REMOVABLE_SUFFIXES = {
    ".odb", ".sim", ".prt", ".res", ".stt", ".mdl", ".pac", ".abq",
    ".sel", ".lck", ".023",
}


def _resolved(path):
    return os.path.normcase(os.path.abspath(os.path.realpath(path)))


def _inside(path, parent):
    path, parent = _resolved(path), _resolved(parent)
    return path.startswith(parent + os.sep)


def _artifact_records(job_directory, allowed_root):
    """List only removable files within an explicitly authorised run folder."""
    job_directory, allowed_root = _resolved(job_directory), _resolved(allowed_root)
    if not _inside(job_directory, allowed_root):
        raise ValueError("CLEANUP_DIRECTORY_OUTSIDE_ALLOWED_ROOT:%s" % job_directory)
    if not os.path.isdir(job_directory):
        return []
    records = []
    for root, directories, files in os.walk(job_directory):
        # Do not traverse symlinked directories into another project location.
        directories[:] = [name for name in directories
                          if not os.path.islink(os.path.join(root, name))]
        for name in files:
            path = os.path.join(root, name)
            if os.path.islink(path):
                continue
            if os.path.splitext(name)[1].lower() not in REMOVABLE_SUFFIXES:
                continue
            records.append({"path": path, "size_bytes": os.path.getsize(path)})
    return sorted(records, key=lambda row: row["path"].lower())


def cleanup_abaqus_job(job_directory, allowed_root, extraction_succeeded):
    """Delete regenerable artifacts only after a caller verified extraction.

    The function does not delete DAT/MSG/STA/LOG/COM/INP, source, manifests,
    CSV, JSON, Markdown, images, or the directory itself.
    """
    if not extraction_succeeded:
        return {"deleted": [], "freed_bytes": 0, "status": "SKIPPED_EXTRACTION_NOT_VERIFIED"}
    records = _artifact_records(job_directory, allowed_root)
    deleted, freed = [], 0
    for record in records:
        os.remove(record["path"])
        deleted.append(record)
        freed += record["size_bytes"]
    return {"deleted": deleted, "freed_bytes": freed, "status": "CLEANED"}


def cleanup_abandoned_abaqus_job(job_directory, allowed_root, diagnostics_preserved):
    """Remove only regenerable artifacts from an explicitly abandoned run.

    This is deliberately separate from ``cleanup_abaqus_job`` so future
    successful-job automation cannot accidentally discard failed diagnostics.
    It retains DAT/MSG/STA/LOG/COM/INP and requires a caller to record that
    those diagnostics were preserved first.
    """
    if not diagnostics_preserved:
        return {"deleted": [], "freed_bytes": 0, "status": "SKIPPED_DIAGNOSTICS_NOT_PRESERVED"}
    records = _artifact_records(job_directory, allowed_root)
    deleted, freed = [], 0
    for record in records:
        os.remove(record["path"])
        deleted.append(record)
        freed += record["size_bytes"]
    return {"deleted": deleted, "freed_bytes": freed, "status": "CLEANED_ABANDONED_DIAGNOSTICS_RETAINED"}


def write_manifest(path, candidates):
    """Write a human-readable, pre-deletion inventory of exact file targets."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Abaqus cleanup manifest before deletion\n\n")
        handle.write("Only explicit, accepted historical run directories are listed.\n\n")
        handle.write("| File | Size (GB) | Reason |\n|---|---:|---|\n")
        for candidate in candidates:
            for record in _artifact_records(candidate["directory"], candidate["allowed_root"]):
                handle.write("| `%s` | %.6f | %s |\n" %
                             (record["path"], record["size_bytes"] / float(1024 ** 3),
                              candidate["reason"]))
