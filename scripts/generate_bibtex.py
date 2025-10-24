import pathlib
import textwrap
import time
import warnings

import requests

import pedpy

zenodo_path = pathlib.Path("docs/source/ZENODO.rst")

search_query = "PedPy"
version = f"v{pedpy.__version__}"
record_id = None
zenodo_record = "If you use *PedPy* in your work, please cite it with the following information from Zenodo.\n\n"


def fetch_data_with_retries(
    url, params=None, headers=None, max_retries=10, wait_time=2
):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            warnings.warn(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(wait_time)
    raise RuntimeError("All attempts to fetch data failed.")


try:
    response = fetch_data_with_retries(
        "https://zenodo.org/api/records",
        params={"q": search_query, "all_versions": True, "sort": "mostrecent"},
    )
    data = response.json()

    # Check for errors
    if "status" in data and data["status"] != 200:
        raise RuntimeError("Not found")
    else:
        # Print available records
        for record in data.get("hits", {}).get("hits", []):
            if (
                "version" in record["metadata"]
                and version == record["metadata"]["version"]
            ):
                record_id = record["id"]

    headers = {"accept": "application/x-bibtex"}
    response = fetch_data_with_retries(
        f"https://zenodo.org/api/records/{record_id}", headers=headers
    )
    response.encoding = "utf-8"

    if response.status_code == 200:
        zenodo_record += (
            ".. code-block:: bibtex\n\n"
            + textwrap.indent(response.text, " " * 4)
            + "\n"
        )
    else:
        raise RuntimeError("Not found")

except Exception as e:
    warnings.warn(f"An error occurred: {e}")

zenodo_record += textwrap.dedent(
    """\

        Information to all versions of PedPy can be found on `Zenodo <https://zenodo.org/doi/10.5281/zenodo.7194992>`_.

        .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7194992.svg
            :target: https://doi.org/10.5281/zenodo.7194992

        To find your installed version of *PedPy*, you can run:

        .. code-block:: bash

            import pedpy
            print(pedpy.__version__)
    """
)

with open(zenodo_path, "w") as f:
    f.write(zenodo_record)

print(zenodo_record)
