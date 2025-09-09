"""Take a candidate catalog and generate a phrosty
lightcurve command file and associated template, science files.

Heaviliy inspired from Lauren Aldoroty's bulk_sne.py
"""

import requests

from astropy.table import Table
import pandas as pd


bands = ['R062', 'Z087', 'Y106', 'J129', 'H158', 'F184', 'K213']

def construct_phrosty_script(oid, ra, dec):
    """
    oid : int
    ra : float, decimal degrees
    dec : float, decimal degrees
    """
    
    templates_filename, science_filename = write_image_list_for_ra_dec(oid, ra, dec)

    oc = "manual"
    ic = "ou2024"

    script = f"""python phrosty/phrosty/pipeline.py \
-c phrosty/examples/perlmutter/phrosty_config.yaml \
--oc {oc} \
--oid {oid} \
-r {ra} \
-d {dec} \
--ic {ic} \
-t {templates_filename} \
-s {science_filename} \
-p 3 \
-w 3 \
"""

    for band in bands:
        this_script = script + f"-b {band} "
        print(this_script)


def get_images_for_ra_dec_mjd(ra, dec,
                              start_mjd, end_mjd,
                              bands=bands,
                              n_templates=10,
                              server_url="https://roman-desc-simdex.lbl.gov",
                              verbose=False):
    """
    oid : int, just used for creating output filename
    ra : float, decimal degrees
    dec : float, decimal degrees
    """

    req = requests.Session()
    result = req.post( f'{server_url}/findromanimages/containing=({ra},{dec})' )
    if result.status_code != 200:
        raise RuntimeError( f"Got status code {result.status_code}\n{result.text}" )
    images = Table.from_pandas(pd.DataFrame(result.json()))
    if verbose:
        print(f"Got {len(images)} images for {ra}, {dec}.")

    templates_filename = f"{oid}_templates.csv"
    science_filename = f"{oid}_science.csv"

    head_str = ['path pointing sca mjd band']

    if len(images) > 0:
        template_counter = {band: 0 for band in bands}
        bands, pointings, scas, mjds = images['filter'], images['pointing'], images['sca'], images['mjd']
        template_strs = []
        science_strs = []
        for band, pointing, sca, mjd in zip(bands, pointings, scas, mjds):
            row_str = f'{band}/{pointing}/Roman_TDS_simple_model_{band}_{pointing}_{sca}.fits.gz {pointing} {sca} {mjd} {band}'
            if mjd <= end_mjd and mjd >= start_mjd:
                science_strs.append(row_str)
            else:
                if template_counter[band] < n_templates:
                    template_strs.append(row_str)
                    template_counter[band] += 1

        science_strs = head_str + science_strs
        template_strs = head_str + template_strs
        temp = '\n'.join(template_strs)
        sci = '\n'.join(science_strs)

    return temp, sci


def write_image_list_for_ra_dec(oid, ra, dec):
    """
    oid : int, just used for creating output filename
    ra : float, decimal degrees
    dec : float, decimal degrees
    """

    templates_filename = f"{oid}_templates.csv"
    science_filename = f"{oid}_science.csv"

    temp, sci = get_images_for_ra_dec_mjd(ra, dec, start_mjd, end_mjd)
    with open(templates_filename, 'w') as f:
        f.write(temp)
    with open(science_filename, 'w') as f:
        f.write(sci)

    return templates_filename, science_filename


if __name__ == "__main__":
    oid = "30"
    ra = 6.455598721735157
    dec = -44.23015158024111
    start_mjd = 62300
    end_mjd = 62500

    construct_phrosty_script(oid, ra, dec)
