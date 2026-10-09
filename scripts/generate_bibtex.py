"""Write ``docs/source/ZENODO.rst`` with the citation information for this build.

The BibTeX entry is fetched from Zenodo by *concept id*, the identifier that
stays the same across all releases, rather than by searching for the project
name. A name search matches titles, descriptions and author names, so it can
return an unrelated record, and it only reads one page of results, which
silently hides older versions once a project has enough of them.

Zenodo being briefly unreachable must not fail the documentation build, so
every failure degrades instead: to the most recent release, and finally to a
note in place of the entry.
"""

import logging
import pathlib
import textwrap

from zenodo_bibtex_exporter import ZenodoBibtexError, get_bibtex

import pedpy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

#: PedPy on Zenodo. This is the concept id, which never changes between releases.
CONCEPT_ID = "7194992"

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
ZENODO_PATH = REPOSITORY_ROOT / "docs" / "source" / "ZENODO.rst"

INTRO = "If you use *PedPy* in your work, please cite it with the following information from Zenodo.\n\n"

TRAILER = textwrap.dedent(
    f"""\

    Information to all versions of PedPy can be found on `Zenodo <https://zenodo.org/doi/10.5281/zenodo.{CONCEPT_ID}>`_.

    .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.{CONCEPT_ID}.svg
        :target: https://doi.org/10.5281/zenodo.{CONCEPT_ID}

    To find your installed version of *PedPy*, you can run:

    .. code-block:: python

        import pedpy
        print(pedpy.__version__)
    """
)


def fetch_bibtex() -> str | None:
    """Return the BibTeX entry to document, or None if Zenodo gave us nothing.

    Prefers the record for the version being built. Development builds carry a
    suffix that no release has, so they fall back to the most recent release.
    """
    version = f"v{pedpy.__version__}"

    try:
        entry = get_bibtex(CONCEPT_ID, version=version)
    except ZenodoBibtexError as error:
        logger.warning("No Zenodo record for %s: %s", version, error)
    else:
        logger.info("Using the Zenodo record for %s.", version)
        return entry

    try:
        entry = get_bibtex(CONCEPT_ID)
    except ZenodoBibtexError as error:
        logger.warning("No citation information available at all: %s", error)
        return None
    else:
        logger.info("Falling back to the most recent release on Zenodo.")
        return entry


def main() -> None:
    """Generate the citation page."""
    entry = fetch_bibtex()

    if entry is None:
        citation = (
            f"Citation information could not be retrieved from Zenodo. "
            f"It is available at https://doi.org/10.5281/zenodo.{CONCEPT_ID}.\n"
        )
    else:
        citation = ".. code-block:: bibtex\n\n" + textwrap.indent(entry, " " * 4)

    ZENODO_PATH.write_text(INTRO + citation + TRAILER, encoding="utf-8")
    logger.info("Wrote %s", ZENODO_PATH)


if __name__ == "__main__":
    main()
